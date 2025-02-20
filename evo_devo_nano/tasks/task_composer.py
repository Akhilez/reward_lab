"""

There will be 3 phases for every step.

Phase 1:
- No grad.
- Loop until all tasks finish phase 1.
- Each iteration
    - gather sequences for batching from all tasks.
    - gather the masks for each task.
    - Forward the batch with padding.
    - cache kv
    - update the task states.
    - Check for termination.
- Figure out what the best action is and step the environment.

Phase 2:
- Gather all the sequences and masks for each task.
- Forward the batch with padding.

Phase 3:
- Compute losses for each task
- Update the model parameters

"""


from evo_devo_nano.interaction import Interaction
from evo_devo_nano.model.model import CraftaxModel


class TaskComposer:
    def __init__(self, model: CraftaxModel, interaction: Interaction):
        self.model = model
        self.interaction = interaction


class NextActionTask:
    def __init__(self):
        pass

    def phase1(self, model, interaction):
        """
        We need the observation's embeddings.

        You need to return the sequence to the model.

        """
        pass
