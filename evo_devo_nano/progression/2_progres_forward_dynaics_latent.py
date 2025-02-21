import torch
from torch.nn import functional as F

from evo_devo_nano.interaction import Interaction, Memory
from evo_devo_nano.model.model import CraftaxModel
from evo_devo_nano.video_io import VideoWriter


def do_it_again():

    video_writer = VideoWriter("/Users/akhildevarashetti/code/reward_lab/evo_devo_nano/progression/2.mp4", h=63, w=63, fps=10)
    recon_writer = VideoWriter("/Users/akhildevarashetti/code/reward_lab/evo_devo_nano/progression/2_recon.mp4", h=63, w=63, fps=10)

    interaction = Interaction()
    memory = Memory(5)
    model = CraftaxModel(h=interaction.h, w=interaction.w, n_actions=interaction.n_actions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ----------------- Recon obs backprop -----------------
    model.train()
    obs = torch.tensor(interaction.obs)  # [64, 64, 3] (0-1)
    video_writer.write_frame((obs * 255).to(torch.uint8).numpy())
    obs = obs.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 64, 64]
    # obs_embeddings, reconstruction = model.obs_encoder(obs, return_reconstruction=True)  # [1, 36, 64], [1, 3, 64, 64]
    # loss_recon = F.mse_loss(reconstruction, obs)
    # recon_writer.write_frame((reconstruction[0].permute(1, 2, 0) * 255).to(torch.uint8).numpy())
    #
    # # Backprop
    # optimizer.zero_grad()
    # loss_recon.backward()
    # optimizer.step()
    #
    # # obs_embeddings = obs_embeddings.detach()
    # # obs_embeddings = model.time_encoder(obs_embeddings, timestep=0)  # [1, 38, 64]

    memory.observations.append(obs)

    for i in range(200):

        action = torch.randint(0, interaction.n_actions, (1,))  # [1,]
        # ------------ Step ---------------
        interaction.step(action[0].item())


        # ---------------- Recon next obs --------------------
        model.train()

        obs = memory.observations[-1]  # [1, 3, 63, 63]

        obs_next = torch.tensor(interaction.obs)  # [63, 63, 3]
        video_writer.write_frame((obs_next * 255).to(torch.uint8).numpy())
        obs_next = obs_next.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 63, 63]
        memory.observations.append(obs_next)

        batch_obs_images = torch.cat([obs, obs_next], dim=0)  # [2, 3, 63, 63]
        batch_obs_embeddings, batch_reconstruction = model.obs_encoder(batch_obs_images, return_reconstruction=True)  # [2, 36, 64], [2, 3, 63, 63]
        obs_embeddings = batch_obs_embeddings[:1]  # [1, 36, 64]
        obs_next_embeddings = batch_obs_embeddings[1:]  # [1, 36, 64]
        recon_writer.write_frame((batch_reconstruction[1].permute(1, 2, 0) * 255).to(torch.uint8).numpy())

        # -------------- Gather embeddings --------------
        obs_embeddings = model.time_encoder(obs_embeddings, timestep=i)  # [1, 38, 64]
        obs_next_embeddings = model.time_encoder(obs_next_embeddings, timestep=i + 1)

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
        bert_out = model.transformer(batch_bert, is_causal=False)  # [2, 38, 64]
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
        # TODO: Create mask
        latent_out = model.transformer(seq_forward_dynamics)  # [2, 76, 64]
        obs2_latent_pred = latent_out[:, -1:]  # [1, 1, 64]

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

        outputs = model.transformer(seq_forward_dynamics)  # [1, 76, 64]

        # -------------- Losses -------------

        loss_recon = F.mse_loss(batch_reconstruction, batch_obs_images)

        obs2_emb_with_grad = outputs[:, -obs_next_embeddings.shape[1]:]  # [1, 37, 64]
        loss_forward_dynamics = F.mse_loss(obs2_emb_with_grad, obs_next_embeddings)

        loss_latent_obs = F.mse_loss(obs2_latent_pred, obs2_latent.detach())

        loss = loss_recon + loss_forward_dynamics + loss_latent_obs

        print({
            "action": action[0].item(),
            # "action_greedy": action_logits.argmax(-1),
            # "action_pred_latent": action_logits_latent.argmax(-1),
            # "action_logits": action_logits,
            # "action_logits_latent": action_logits_latent,
            # "reward_full_state": reward_full_state,
            # "reward_latent": reward_latent,
            "loss_recon": loss_recon.item(),
            "loss_forward_dynamics": loss_forward_dynamics.item(),
            # "loss_inverse_dynamics": loss_inverse_dynamics,
            # "loss_action_latent": loss_action_latent,
            # "loss_policy": loss_policy,
            "loss_latent_obs": loss_latent_obs.item(),
            "loss": loss.item(),
        })

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    video_writer.release()
    recon_writer.release()




if __name__ == "__main__":
    # do_it()
    do_it_again()

