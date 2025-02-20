import torch
from torch.nn import functional as F

from evo_devo_nano.interaction import Interaction, Memory
from evo_devo_nano.model.model import CraftaxModel


def do_it_again():
    interaction = Interaction()
    memory = Memory(5)
    model = CraftaxModel(h=interaction.h, w=interaction.w, n_actions=interaction.n_actions)

    model.train()
    obs = torch.tensor(interaction.obs)  # [64, 64, 3] (0-1)
    obs = obs.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 64, 64]
    obs_embeddings, reconstruction = model.obs_encoder(obs, return_reconstruction=True)  # [1, 36, 64], [1, 3, 64, 64]
    loss_recon = F.mse_loss(reconstruction, obs)
    # TODO: backprop

    obs_embeddings = obs_embeddings.detach()
    # prefix = model.vocab["next_state_key"]  # [64,]
    # prefix = prefix.unsqueeze(0)  # [1, 64]
    # obs_embeddings = torch.cat([obs_embeddings, prefix], dim=1)  # [1, 37, 64]
    obs_embeddings = model.time_encoder(obs_embeddings, timestep=0)  # [1, 38, 64]

    memory.observations.append(obs_embeddings)

    for i in range(5):
        with torch.no_grad():
            obs_embeddings = memory.observations[-1]  # [1, 37, 64]
            action_key = model.vocab["action_key"]  # [64,]
            inputs = torch.cat([obs_embeddings, action_key.view(1, 1, -1)], dim=1)  # [1, 38, 64]

            model.transformer.eval()
            output_embeddings = model.transformer(inputs)  # [1, 38, 64]
            action_logits = model.classifier(output_embeddings[:, -1, :])
            action_logits = action_logits[:, torch.where(model.vocab.actions_mask)[0]]
            action_logits = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_logits, num_samples=1)[0]

        interaction.step(action[0].item())

        # Phase 2
        model.train()
        obs_next = torch.tensor(interaction.obs)  # [64, 64, 3]
        obs_next = obs_next.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 63, 63]
        obs_next_embeddings, reconstruction_next = model.obs_encoder(obs_next, return_reconstruction=True)
        loss_recon = F.mse_loss(reconstruction_next, obs_next)

        obs_next_embeddings = obs_next_embeddings.detach()
        obs_next_embeddings = model.time_encoder(obs_next_embeddings, timestep=i+1)
        memory.observations.append(obs_next_embeddings)

        """
        We have obs, a and onext.
        now, we can create a few sequences to forward.
        Primarily, we need reward from inverse and forward dynamics like ICM.
        
        Once we have the reward, we can create more sequences to forward. This is phase 3.
        
        Finally, we can compute the loss and update the model. Call it phase 4.
        
        
        """







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
    do_it()


