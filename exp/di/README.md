
# Deep Intelligence

So here's the game plan

The journey should take me from barely able to design a basic 3D model in MJCF xml
to creating environments and models that incorporate brain inspired modules and evolution.

So how do we get there? Baby steps. Baby steps indeed.
I learned from learning ASL that learning for 30 hours over a month is very efficient learning 
than learning for 30 hours in a week. Let's apply a similar method here in that I want to learn it deep even if
it takes long.

On very high level, I will first need to create a framework to 
create a very basic locomotion RL env for a basic robot and train to learn the actions.
Once point A to B framework is established, I can now make several parallel experiments.
One being phenotype modifications for evolution. One where the target is moving.
One for online RL for real world-like training.
One for adding brain-inspired modules like CPG, place cells, etc.
Maybe multi-agent learning.

-----------------
Part 1: Building the framework

  - [x] ~~The biggest challenge would be to build the initial framework.
I'll have to learn from building and understanding robots in MuJoCo GUI at first.
Then simulating the same in python, maybe create graphs for the positions.
Now add a few actuators to make it interactive. Apply those forces in python.~~

  - [x] ~~Now, re-create the same with PyMJCF to basically create a robot architecture on the fly. Randomize joint locations.
Okay, now create some tiny robots that can walk. Randomize the actions and plot locomotion.
Cool, it's time to create some target and distance metric for reward.~~

  - [x] ~~Once you have this setup, it's time to graduate to actual RL stuff.
Learn the composer framework and create the basic locomotion env.~~

  - [x] ~~Get the env to work with a basic RL algorithm.~~

With this, we end part 1. Fuck yeah!


-----------------
Part 2: Evolution
Now that we have an RL env, can we manipulate the genotype of the robots and run a large evolutionary algorithm?


-----------------
Part 3: Brain Inspired Modules


-----------------
Part 4: First baby step of DI

Now we can apply both brain inspired online learning and evolve. Throw in developmental learning in there.
Now are we in the right path towards DI?

-----------------