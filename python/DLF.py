from BASE_MODEL import BASE_RNN

# default parameter
FEATURE_SIZE = 16  # dataset input fields count
MAX_DEN = 580000  # max input data dimension
EMB_DIM = 32
BATCH_SIZE = 128
MAX_SEQ_LEN = 330
TRAING_STEPS = 2000
STATE_SIZE = 128
GRAD_CLIP = 5.0
L2_NORM = 0.001
ADD_TIME = True
ALPHA = 1.2  # coefficient for cross entropy
BETA = 0.2  # coefficient for anlp
# input_file = "2259"  # toy dataset

input_files = ['3386', '3427', '3476', '1458', '2997', '2259', '2261', '2821']

print("Please input learning rate. ex. 0.0001")
LR = float(float(input()))
LR_ANLP = LR

for input_file in input_files:
    RUNNING_MODEL = BASE_RNN(EMB_DIM=EMB_DIM,
                             FEATURE_SIZE=FEATURE_SIZE,
                             BATCH_SIZE=BATCH_SIZE,
                             MAX_DEN=MAX_DEN,
                             MAX_SEQ_LEN=MAX_SEQ_LEN,
                             TRAING_STEPS=TRAING_STEPS,
                             STATE_SIZE=STATE_SIZE,
                             LR=LR,
                             GRAD_CLIP=GRAD_CLIP,
                             L2_NORM=L2_NORM,
                             INPUT_FILE=input_file,
                             ALPHA=ALPHA,
                             BETA=BETA,
                             ADD_TIME_FEATURE=ADD_TIME,
                             FIND_PARAMETER=False,
                             ANLP_LR=LR,
                             DNN_MODEL=False,
                             DISCOUNT=1,
                             ONLY_TRAIN_ANLP=False,
                             LOG_PREFIX="dlf")
    RUNNING_MODEL.create_graph()
    RUNNING_MODEL.run_model()
