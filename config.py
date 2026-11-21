
# Don't change this.
# This config fits under 200k trainable parameters.
# I tried a few other configurations, more blocks and less convolutions, but this one is about the strongest.
# We can't go less than 3 blocks because then its too shallow.
# If we increase the blocks (more layers). Then each channel doesn't have enough information to pass to the next layer. making it pointless.
BLOCKS = 3
CHANNELS = 72
HEADS = 2




# Set this to 0 for continous training. Otherwise it will simply train on this amount of games then stop.
# Preferably keep this at increments of PARALLEL * 4. This syncs up the games trained and when we make backup checkpoints.
GAMES = 0

# If you're just going to do a super short training run, then comment out the above and uncomment the below.
# Change the 1 to however many checkpoints you want to train and then stop.
# GAMES = PARALLEL * 4 * 1


# Self play simulations.
# More simulations = stronger learning target, slower training
# Less simulations = weaker learning target, faster training
SIMULATIONS = 1600

# Don't change this.
# This is the playout cap randomization in KataGo. Only 25% percent of moves get the full search.
# The other 75% of moves get a sixth of the simulations. We don't train of the short simulation moves, that would weaken the strength of the bot.
# This significantly speeds up the amount of games we can train on. 
FULL_SEARCH_FRACTION = 0.25
FAST_SIMULATIONS = SIMULATIONS // 6

# Games to play in parallel on gpu. Spread across the workers. 
# This is dependent on your pc. For me 512 games on 16 workers was good for me. Maybe not for you.
# If you want to change it, start a timer on your phone and see how fast it trains.
# If you dont have enough ram then decrease PARALLEL. Keep PARALLEL as a power of 2. So try 256, 128 etc.
# Changing workers barely effects ram, it just affects training speed depending on your machine.
PARALLEL = 512
WORKERS = 16

# Don't change this. If you want more training per game raise REPLAY_SAMPLE instead.
EPOCHS = 1

# How many positions go into each training step. REPLAY_SAMPLE gets split into batches of this size.
BATCH_SIZE = 128

# I wouldn't change these. just the default learning rate.
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.1

# How many positions we train on after every finished game, picked at random from the buffer.
# this is good I think. Trying to keep everything at multiples of 2 for cleanliness.
REPLAY_SAMPLE = 128

# Starting size of the replay buffer. Each game adds about 14 positions, so 16 per game always fits a whole batch of games.
# It grows on its own as training goes on, KataGos method of growing the buffer over time.
REPLAY_BUFFER_SIZE = PARALLEL * 16

# Where the model gets saved. The replay buffer gets saved right next to it.
# Change the name if you want to start a fresh run.
CHECKPOINT = "checkpoints/reversi_model.pt"

# Every this many games we keep a copy of the model with the game count in the name, and save the replay buffer.
# Set it to 0 to turn backups off.
BACKUP_EVERY = PARALLEL * 4

# Random seed. Doesn't really matter what it is.
SEED = 42
