# Hyperparameters
LATENT_DIM = 32
D_MODEL = 1024
RATE = 0.1  # Specifically, the dropout rate
EPOCHS = 100
BATCH_SIZE = 4
RASTER_LOSS_WEIGHT = 15000.0

# Vectorization sub-model weights
VECTOR_LOSS_WEIGHT_COMMAND = 1.0  # Keep this at 1, normalize others against it
VECTOR_RASTERIZATION_LOSS_WEIGHT = 0.1
VECTOR_LOSS_WEIGHT_COORD = 0.1
CONTOUR_COUNT_WEIGHT = 0.0
NODE_COUNT_WEIGHT = 0.00
HANDLE_SMOOTHNESS_WEIGHT = 0.0
RASTER_LOSS_CUTOFF = 0.02  # Only apply raster loss if less than this value

EOS_SOFTMAX_TEMPERATURE = 0.1
HUBER_DELTA = 3.0
LOSS_IMAGE_SIZE = 256  # Size to rasterize images to for raster loss calculation

LEARNING_RATE = 3e-3
FINAL_LEARNING_RATE = 5e-7

GEN_IMAGE_SIZE = (512, 512)
RASTER_IMG_SIZE = GEN_IMAGE_SIZE[0]
STYLE_IMAGE_SIZE = (168, 40)
MAX_COMMANDS = 50
LIMIT = 0  # Limit the number of fonts to process for testing

# New tokenization hyperparameters
QUANTIZATION_BIN_SIZE = 10
COORD_RANGE = (-1000, 1000)
MAX_SEQUENCE_LENGTH = MAX_COMMANDS + 2  # +2 for SOS and EOS tokens

ALPHABET = ["a", "d", "h", "e", "s", "i", "o", "n", "t"]
# # While pre-training the vectorizer, shove as many glyphs in as we can
import string

ALPHABET = list(string.ascii_lowercase + string.ascii_uppercase + string.digits)
BASE_DIR = "/Users/simon/others-repos/fonts/ofl"
# BASE_DIR = "/mnt/experiments/fonts/ofl"

NUM_GLYPHS = len(ALPHABET)
