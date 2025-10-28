# Hyperparameters
LATENT_DIM = 32
D_MODEL = 256
RATE = 0.75  # Specifically, the dropout rate
EPOCHS = 100
BATCH_SIZE = 128
RASTER_LOSS_WEIGHT = 15000.0

# Vectorization sub-model weights
VECTOR_LOSS_WEIGHT_COMMAND = 0.5  # Keep this at 1, normalize others against it
VECTOR_RASTERIZATION_LOSS_WEIGHT = 0.1
VECTOR_LOSS_WEIGHT_COORD = 0.001
CONTOUR_COUNT_WEIGHT = 5.0
NODE_COUNT_WEIGHT = 0.08
HANDLE_SMOOTHNESS_WEIGHT = 0.0
SIGNED_AREA_WEIGHT = 1e-6
RASTER_BLACK_PIXEL_WEIGHT = 10.0
RASTER_LOSS_CUTOFF = 0.05  # Only apply raster loss if raster loss less than this value

# Turn stuff off again
HANDLE_SMOOTHNESS_WEIGHT = 0
VECTOR_RASTERIZATION_LOSS_WEIGHT = 0
CONTOUR_COUNT_WEIGHT = 0
NODE_COUNT_WEIGHT = 0

EOS_SOFTMAX_TEMPERATURE = 0.1
CONTOUR_LOSS_TEMPERATURE = 2.0
HUBER_DELTA = 3.0
LOSS_IMAGE_SIZE = 256  # Size to rasterize images to for raster loss calculation

LEARNING_RATE = 1e-3 * (256 / BATCH_SIZE)
FINAL_LEARNING_RATE = 1e-6

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

ALPHABET = "o{}[]"

BASE_DIR = "/home/simon/others-repos/fonts/ofl"
# BASE_DIR = "/mnt/experiments/fonts/ofl"

NUM_GLYPHS = len(ALPHABET)
