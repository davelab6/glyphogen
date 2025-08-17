# Hyperparameters
NUM_GLYPHS = 32
LATENT_DIM = 32
NUM_TRANSFORMER_LAYERS = 4
D_MODEL = 256
NUM_HEADS = 8
DFF = D_MODEL * 4
RATE = 0.1  # Specifically, the dropout rate
EPOCHS = 100
BATCH_SIZE = 16
RASTER_LOSS_WEIGHT = 100.0
VECTOR_LOSS_WEIGHT_COMMAND = 5000.0
VECTOR_LOSS_WEIGHT_COORD = 1.0
LEARNING_RATE = 1.2e-4
GEN_IMAGE_SIZE = (512, 512)
STYLE_IMAGE_SIZE = (168, 40)
MAX_COMMANDS = 150
LIMIT = 10  # Limit the number of fonts to process for testing

# New tokenization hyperparameters
QUANTIZATION_BIN_SIZE = 10
COORD_RANGE = (-1000, 1000)
MAX_SEQUENCE_LENGTH = (MAX_COMMANDS * 7) + 1

ALPHABET = ["a", "d", "h", "e", "s", "i", "o", "n", "t"]
BASE_DIR = "/Users/simon/others-repos/fonts/ofl/"