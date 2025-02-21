"""

- get initial obs
- autoencoder backprop
- push obs embeddings to memory.
- Loop:
    - Action to step:
        - get latest obs embeddings from memory
        - Predict next action
        - Step
    - Gather obs2 emb
        - auto-encoder backprop
    - Reward by forward and inverse dynamics
        - Full state
            - no grad predict next obs embeddings
            - Calculate reward1 from obs2_emb and obs2_emb_pred
        - Latent state
            - BERT & latent embeddings of obs1 & obs2
            - predict next action from latent
            - predict latent obs2 from obs2 & action.
            - Calculate reward2 from latent_obs2 and latent_obs2_pred
    - Forward sequences
    - Loss computation & backward


"""
import torch
from torch.nn import functional as F

from evo_devo_nano.interaction import Interaction, Memory
from evo_devo_nano.model.model import CraftaxModel


def do_it_again():
    temperature = 100  # lower temperature means more greedy sampling

    interaction = Interaction()
    memory = Memory(5)
    model = CraftaxModel(h=interaction.h, w=interaction.w, n_actions=interaction.n_actions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ----------------- Recon obs backprop -----------------
    model.train()
    obs = torch.tensor(interaction.obs)  # [64, 64, 3] (0-1)
    obs = obs.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 64, 64]
    obs_embeddings, reconstruction = model.obs_encoder(obs, return_reconstruction=True)  # [1, 36, 64], [1, 3, 64, 64]
    loss_recon = F.mse_loss(reconstruction, obs)

    # Backprop
    optimizer.zero_grad()
    loss_recon.backward()
    optimizer.step()

    obs_embeddings = obs_embeddings.detach()
    # prefix = model.vocab["next_state_key"]  # [64,]
    # prefix = prefix.unsqueeze(0)  # [1, 64]
    # obs_embeddings = torch.cat([obs_embeddings, prefix], dim=1)  # [1, 37, 64]
    obs_embeddings = model.time_encoder(obs_embeddings, timestep=0)  # [1, 38, 64]

    memory.observations.append(obs_embeddings)

    for i in range(10):
        obs_embeddings = memory.observations[-1]  # [1, 37, 64]
        action_key = model.vocab["action_key"]  # int
        action_key = model.embeddings(torch.tensor(action_key).unsqueeze(0))  # [1, 64]
        inputs = torch.cat([obs_embeddings, action_key.view(1, 1, -1)], dim=1)  # [1, 38, 64]

        # ----------------- Predict action -----------------
        model.transformer.eval()
        output_embeddings = model.transformer(inputs)  # [1, 38, 64]
        action_logits = model.classifier(output_embeddings[:, -1, :])  # [1, n_classes]
        action_logits = action_logits[:, torch.where(model.vocab.actions_mask)[0]]  # [1, n_actions]
        action_logits = torch.softmax(action_logits / temperature, dim=-1)  # [1, n_actions]
        action = torch.multinomial(action_logits, num_samples=1)[0]  # [1]

        # ------------ Step ---------------
        interaction.step(action[0].item())


        # ---------------- Recon next obs --------------------
        model.train()
        obs_next = torch.tensor(interaction.obs)  # [64, 64, 3]
        obs_next = obs_next.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 63, 63]
        obs_next_embeddings, reconstruction_next = model.obs_encoder(obs_next, return_reconstruction=True)

        # -------------- Gather embeddings --------------
        obs_next_embeddings = obs_next_embeddings.detach()
        obs_next_embeddings = model.time_encoder(obs_next_embeddings, timestep=i + 1)
        memory.observations.append(obs_next_embeddings)

        action_embedding = model.embeddings(model.vocab.action_indices[action])  # [1, 64]
        action_embedding = action_embedding.unsqueeze(1)  # [1, 1, 64]
        memory.actions.append(action_embedding)

        next_state_key = model.vocab["next_state_key"]  # int
        next_state_key = model.embeddings(torch.tensor(next_state_key).unsqueeze(0))  # [1, 64]
        next_state_key = next_state_key.unsqueeze(0)  # [1, 1, 64]

        action_key = model.vocab["action_key"]  # int
        action_key = model.embeddings(torch.tensor(action_key).unsqueeze(0))  # [1, 64]
        action_key = action_key.view(1, 1, -1)  # [1, 1, 64]

        next_latent_obs_key = model.vocab["next_latent_obs_key"]  # int
        next_latent_obs_key = model.embeddings(torch.tensor(next_latent_obs_key).unsqueeze(0))  # [1, 64]
        next_latent_obs_key = next_latent_obs_key.view(1, 1, -1)  # [1, 1, 64]

        # -------------- Predict next state -------------

        with torch.no_grad():
            seq_forward_dynamics = torch.cat([
                next_state_key,
                obs_embeddings,
                action_key,
                action_embedding,
                next_state_key,
            ], dim=1)  # [1, 76, 64]
            # Iterate forward with KV caching until len(obs_next_embeddings) are predicted.
            for j in range(obs_next_embeddings.shape[1]):
                outputs = model.transformer(seq_forward_dynamics)  # [1, 76, 64]
                next_obs_embedding = outputs[:, -1:]  # [1, 1, 64]
                seq_forward_dynamics = torch.cat([seq_forward_dynamics, next_obs_embedding], dim=1)  # [1, 77, 64]

        obs_next_pred = seq_forward_dynamics[:, -obs_next_embeddings.shape[1]:]  # [1, 37, 64]

        # -------------- BERT & latent obs -------------
        model.train()
        seq_obs1 = torch.cat([
            next_state_key,
            obs_embeddings,
        ], dim=1)  # [1, 38, 64]
        seq_obs2 = torch.cat([
            next_state_key,
            obs_next_embeddings,
        ], dim=1)  # [1, 38, 64]
        batch_bert = torch.cat([seq_obs1, seq_obs2], dim=0)  # [2, 38, 64]
        # TODO: Create BERT mask
        bert_out = model.transformer(batch_bert)  # [2, 38, 64]
        obs1_latent = bert_out[:1, :1]  # [1, 1, 64]
        obs2_latent = bert_out[1:, :1]  # [1, 1, 64]

        # -------------- Predict next latent obs ------------
        seq_forward_dynamics = torch.cat([
            next_latent_obs_key,
            obs1_latent,
            action_key,
            action_embedding,
            next_latent_obs_key,
            # obs2_latent,
        ], dim=1)  # [1, 76, 64]
        seq_inverse_dynamics = torch.cat([
            next_latent_obs_key,
            obs1_latent,
            next_latent_obs_key,
            obs2_latent,
            action_key,
            # action_embedding,
        ], dim=1)
        batch_latent = torch.cat([seq_forward_dynamics, seq_inverse_dynamics], dim=0)  # [2, 76, 64]
        # TODO: Create mask
        latent_out = model.transformer(batch_latent)  # [2, 76, 64]

        action_embedding_pred = latent_out[1:, -1:]  # [1, 1, 64]
        action_logits_latent = model.classifier(action_embedding_pred)  # [1, 1, n_classes]
        action_logits_latent = action_logits_latent[:, :, torch.where(model.vocab.actions_mask)[0]]  # [1, 1, n_actions]
        action_logits_latent = torch.softmax(action_logits_latent, dim=-1)  # [1, 1, n_actions]

        obs2_latent_pred = latent_out[:1, -1:]  # [1, 1, 64]

        loss_latent_obs = F.mse_loss(obs2_latent_pred, obs2_latent)

        # --------------- Compute reward --------------
        reward_full_state = F.mse_loss(obs_next_pred, obs_next_embeddings)
        reward_latent = loss_latent_obs.detach()

        # --------------- Training -----------------

        seq_forward_dynamics = torch.cat([
            next_state_key,
            obs_embeddings,
            action_key,
            action_embedding,
            next_state_key,
            obs_next_embeddings[:, :-1],  # --> next emb
        ], dim=1)  # [1, 76, 64]
        # TODO: Create mask

        seq_inverse_dynamics = torch.cat([
            next_state_key,
            obs_embeddings,
            next_state_key,
            obs_next_embeddings,
            action_key,  # --> action_embedding
        ], dim=1)  # [1, 76, 64]
        # TODO: Create mask

        # TODO: Create padding
        batch = torch.cat([seq_forward_dynamics, seq_inverse_dynamics], dim=0)  # [2, 76, 64]

        outputs = model.transformer(batch)  # [2, 76, 64]

        # TODO: Predict reward

        # -------------- Losses -------------

        loss_recon = F.mse_loss(reconstruction_next, obs_next)
        loss_action_latent = F.cross_entropy(action_logits_latent.squeeze(1), action)
        # loss_latent_obs

        obs2_emb_with_grad = outputs[0, -obs_next_embeddings.shape[1]:]  # [1, 37, 64]
        loss_forward_dynamics = F.mse_loss(obs2_emb_with_grad, obs_next_embeddings)

        action_pred_with_grad = outputs[1, -1:]  # [1, 64]
        action_logits_pred = model.classifier(action_pred_with_grad)  # [1, n_classes]
        action_logits_pred = action_logits_pred[:, torch.where(model.vocab.actions_mask)[0]]  # [1, n_actions]
        action_logits_pred = torch.softmax(action_logits_pred, dim=-1)  # [1, n_actions]
        loss_inverse_dynamics = F.cross_entropy(action_logits_pred, action)

        # Loss for action prediction
        # - log(action_logits[action]) * reward
        loss_policy = -torch.log(action_logits[:, action]) * (reward_full_state + reward_latent)
        loss_policy = loss_policy.mean()

        loss = loss_recon + loss_forward_dynamics + loss_inverse_dynamics + loss_action_latent + loss_policy + loss_latent_obs

        print({
            "action": action,
            "action_greedy": action_logits.argmax(-1),
            "action_pred_latent": action_logits_latent.argmax(-1),
            "action_logits": action_logits,
            "action_logits_latent": action_logits_latent,
            "reward_full_state": reward_full_state,
            "reward_latent": reward_latent,
            "loss_recon": loss_recon,
            "loss_forward_dynamics": loss_forward_dynamics,
            "loss_inverse_dynamics": loss_inverse_dynamics,
            "loss_action_latent": loss_action_latent,
            "loss_policy": loss_policy,
            "loss_latent_obs": loss_latent_obs,
            "loss": loss,
        })

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def do_it():
    interaction = Interaction()
    memory = Memory(5)
    model = CraftaxModel(h=interaction.h, w=interaction.w, n_actions=interaction.n_actions)

    """
    We need to get the best action for the current observation.
    For that, we first get the embeddings of observation.
    Create a sequence for transformer and sample from the actions.
    """
    obs = torch.tensor(interaction.obs)  # [64, 64, 3] (0-1)
    print(obs.shape, obs.min(), obs.max())

    obs = obs.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 64, 64]

    obs_embeddings, reconstruction = model.obs_encoder(obs, return_reconstruction=True)
    print(obs_embeddings.shape, reconstruction.shape)

    embeddings_with_time = model.time_encoder(obs_embeddings, timestep=0)
    print(embeddings_with_time.shape)

    action_key_index = model.vocab["action_key"]
    print(action_key_index)

    action_key_embeddings = model.embeddings(torch.tensor(action_key_index).unsqueeze(0))  # [1, 64]
    action_key_embeddings = action_key_embeddings.unsqueeze(1)  # [1, 1, 64]
    print(action_key_embeddings.shape)

    embeddings = torch.cat([embeddings_with_time, action_key_embeddings], dim=1)
    print(embeddings.shape)
    memory.observations.append(embeddings)

    output_embeddings = model.transformer(embeddings)
    print(output_embeddings.shape)

    # Last token is the action key.
    action_logits = model.classifier(output_embeddings[:, -1, :])
    # Filter with mask
    action_logits = action_logits[:, torch.where(model.vocab.actions_mask)[0]]
    print(action_logits.shape)

    action_logits = torch.softmax(action_logits, dim=-1)
    print(action_logits)

    action = torch.multinomial(action_logits, num_samples=1)[0]
    print(action)

    action_embedding = model.embeddings(model.vocab.action_indices[action])
    print(action_embedding.shape)

    memory.actions.append(action_embedding)

    # Step the environment
    interaction.step(action[0].item())

    obs_next = torch.tensor(interaction.obs)  # [64, 64, 3] (0-1)
    print(obs_next.shape, obs_next.min(), obs_next.max())

    obs_next = obs_next.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 64, 64]

    obs_next_embeddings, reconstruction_next = model.obs_encoder(obs_next, return_reconstruction=True)
    print(obs_next_embeddings.shape, reconstruction_next.shape)

    embeddings_next_with_time = model.time_encoder(obs_next_embeddings, timestep=1)
    print(embeddings_next_with_time.shape)

    memory.observations.append(embeddings_next_with_time)








if __name__ == "__main__":
    # do_it()
    do_it_again()

