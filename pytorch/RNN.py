# RNN FOOTBALL QUESTION-ANSWER CLASSIFIER
# PyTorch

# 1. IMPORT LIBRARIES

import re
import random
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# 2. HYPERPARAMETERS / CONFIGURATION

# Path to our CSV dataset.
CSV_PATH = "football_qa_dataset.csv"


# Every word will be represented by a vector containing
# 50 numerical values.
# Example:
# "football" -> [0.12, -0.42, 0.73, ..., 0.18]
# There will be exactly 50 numbers in this vector.
EMBEDDING_DIM = 50


# Number of values maintained by the RNN hidden state.
# The RNN will maintain 128 values as its internal
# representation of the sequence.
HIDDEN_SIZE = 128


# Number of stacked RNN layers.
# We are using:
# RNN Layer 1
#      ↓
# RNN Layer 2
# instead of using only one RNN layer.
NUM_LAYERS = 2


# Number of questions processed together during one
# training iteration.
# If BATCH_SIZE = 32, the model processes approximately
# 32 questions before updating its weights.
BATCH_SIZE = 32


# Number of times the entire training dataset is passed
# through the model.
EPOCHS = 50



# Controls how large the parameter updates are during
# optimization.
# A value of 0.001 is a common starting point for Adam.
LEARNING_RATE = 0.001


# Special token used to fill shorter questions so that
# questions inside the same batch have the same length.
PAD_TOKEN = "<PAD>"


# Special token used when a word appears during prediction
# that was not present in our training vocabulary.
UNK_TOKEN = "<UNK>"


# Random seed makes our experiment more reproducible.
SEED = 42


# 3. REPRODUCIBILITY

# Python's random module will use the same random sequence
# every time we run the program.
random.seed(SEED)


# PyTorch will also use the same random initialization
# when possible.
torch.manual_seed(SEED)

# 4. LOAD DATASET

# pandas reads the CSV file and creates a DataFrame.
# Our DataFrame should look approximately like:
# intent_id | category | question | answer
# -------------------------------------------------------
# 53        | world_cup | Name... | West Germany...
# 198       | managers  | Which...| Claudio Ranieri...
df = pd.read_csv(CSV_PATH)


# We only need these four columns.
# dropna() removes rows where one of these columns contains
# a missing value.
df = df[
    [
        "intent_id",
        "category",
        "question",
        "answer"
    ]
].dropna()


# Make sure these columns are treated as strings.
# This prevents unexpected datatype problems later.
df["question"] = df["question"].astype(str)
df["category"] = df["category"].astype(str)
df["answer"] = df["answer"].astype(str)


# 5. TEXT TOKENIZATION

def tokenize(text):
    """
    Convert a question from raw text into individual words.

    Example:

    Input:
        "Which club won the Champions League?"

    Output:
        ["which", "club", "won", "the",
         "champions", "league"]
    """

    # Convert all characters to lowercase.
    # This means:
    # "Club"
    # "club"
    # "CLUB"
    # are treated as the same word.
    text = text.lower()


    # Remove characters other than:
    # a-z
    # 0-9
    # whitespace
    # Therefore:
    # "League?"
    # becomes:
    # "league"
    text = re.sub(r"[^a-z0-9\s]", "", text)


    # split() separates the sentence wherever whitespace
    # occurs.
    # "which club won"
    # becomes:
    # ["which", "club", "won"]
    return text.split()


# 6. BUILD WORD VOCABULARY

# The vocabulary converts words into integer IDs.
# We reserve:
# 0 -> <PAD>
# 1 -> <UNK>
# because these two values have special meanings.
word_to_index = {
    PAD_TOKEN: 0,
    UNK_TOKEN: 1
}


# This loop goes through every question in our dataset.
for question in df["question"]:
    # Tokenize the current question.
    # Example:
    # "Which club won the World Cup?"
    # becomes:
    # ["which", "club", "won", "the", "world", "cup"]
    for word in tokenize(question):
        # If the word does not already exist in our
        # vocabulary, add it.
        if word not in word_to_index:
            # len(word_to_index) gives us the next available
            # integer ID.
            # Example:
            # If the dictionary currently contains 100 words,
            # the next word receives index 100.
            word_to_index[word] = len(word_to_index)


# 7. BUILD INTENT MAPPING

# Get every unique intent ID from the dataset.
# Example:
# [53, 198, 244, 56, 44]
# might become:
# [44, 53, 56, 198, 244]
# after sorting.
intent_ids = sorted(
    df["intent_id"].unique().tolist()
)


# PyTorch CrossEntropyLoss expects class labels such as:
# 0, 1, 2, 3, ...
# Our original intent IDs might look like:
# 53, 198, 244, ...
# Therefore we create a mapping.
intent_to_index = {
    intent_id: index
    for index, intent_id in enumerate(intent_ids)
}


# We also create the reverse mapping.
# Example:
# intent_to_index:
# 198 -> 5
# index_to_intent:
# 5 -> 198
# We will need this when making predictions later.
index_to_intent = {
    index: intent_id
    for intent_id, index in intent_to_index.items()
}

# 8. BUILD CATEGORY MAPPING
# ------------------------------------------------------------
# Get all unique categories.
# Example:
# ["world_cup", "managers", "rules", "league"]
category_names = sorted(
    df["category"].unique().tolist()
)

# Convert category names into numerical class IDs.
# Example:
# "league"    -> 0
# "managers"  -> 1
# "rules"     -> 2
# "world_cup" -> 3
category_to_index = {
    category: index
    for index, category in enumerate(category_names)
}


# Reverse category mapping.
# This will be useful later when displaying predictions.
index_to_category = {
    index: category
    for category, index in category_to_index.items()
}

# 9. CONVERT QUESTION INTO INTEGER SEQUENCE

def encode_question(question):
    """
    Convert a question from words into vocabulary indices.
    Example:
    "which club won"
    might become:
    [12, 43, 71]
    """

    # Convert the question into individual words.
    tokens = tokenize(question)


    # If the question somehow becomes empty after
    # preprocessing, use the unknown token.
    if len(tokens) == 0:
        tokens = [UNK_TOKEN]


    # Convert every word into its integer vocabulary index.
    # .get() means:
    # If the word exists:
    #     return its index
    # Otherwise:
    #     return the <UNK> index.
    return [
        word_to_index.get(
            word,
            word_to_index[UNK_TOKEN]
        )
        for word in tokens
    ]


# Apply encode_question() to every question in the dataset.
# A new column is created:
# encoded_question
# Example:
# "which club won the league"
# becomes:
# [12, 43, 71, 5, 92]
df["encoded_question"] = df["question"].apply(
    encode_question
)


# 10. CONVERT INTENTS AND CATEGORIES INTO CLASS INDICES

# Convert original intent IDs into sequential class indices.
# Example:
# intent_id = 198
# might become:
# intent_index = 5
df["intent_index"] = df["intent_id"].map(
    intent_to_index
)


# Convert category names into numerical class indices.
# Example:
# "managers"
# might become:
# category_index = 2
df["category_index"] = df["category"].map(
    category_to_index
)


# 11. TRAIN / TEST SPLIT

# Split the dataset into:
# 80% training data
# 20% testing data
# random_state ensures that the same split is generated
# when the program is run again.
# stratify makes the split try to preserve the distribution
# of intent IDs between training and testing.
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["intent_id"]
)


# 12. CUSTOM PYTORCH DATASET

class FootballDataset(Dataset):
    """
    Custom Dataset for our football questions.
    Every item returned by this Dataset contains:
        question
        intent
        category
    """
    def __init__(self, dataframe):
        """
        Constructor.
        This runs when we create:
            FootballDataset(train_df)
        or:
            FootballDataset(test_df)
        """
        
        # Store encoded questions.
        self.questions = dataframe[
            "encoded_question"
        ].tolist()


        # Store intent class indices.
        self.intents = dataframe[
            "intent_index"
        ].tolist()


        # Store category class indices.
        self.categories = dataframe[
            "category_index"
        ].tolist()


    def __len__(self):
        """
        Return the total number of examples.
        DataLoader uses this to know how many examples
        exist in the Dataset.
        """
        
        return len(self.questions)


    def __getitem__(self, index):
        """
        Return one training example.
        For example:
        question:
            [12, 43, 71, 8]
        intent:
            5
        category:
            2
        """

        # Convert the question's integer sequence into
        # a PyTorch tensor.
        # LongTensor is required because these numbers
        # are going to be used as indices by nn.Embedding.
        question = torch.tensor(
            self.questions[index],
            dtype=torch.long
        )

        # Convert intent into a LongTensor.
        intent = torch.tensor(
            self.intents[index],
            dtype=torch.long
        )


        # Convert category into a LongTensor.
        category = torch.tensor(
            self.categories[index],
            dtype=torch.long
        )


        # Return all three values.
        return question, intent, category


# 13. COLLATE FUNCTION

def collate_fn(batch):
    """
    Prepare a batch of questions.
    The important problem here is that questions have
    different lengths.
    Example:
        [12, 43, 71]
        [21, 73, 17, 8, 9]
        [7, 19]

    A normal tensor cannot directly store these because
    their lengths are different.
    We therefore pad them.
    """

    # Separate the three parts of every example.
    # zip(*batch) effectively changes:
    # [
    #   (question1, intent1, category1),
    #   (question2, intent2, category2)
    # ]
    # into:
    # questions
    # intents
    # categories
    questions, intents, categories = zip(*batch)


    # Find the length of every question.
    # Example:
    # [3, 5, 2]
    # means:
    # question 1 has 3 words
    # question 2 has 5 words
    # question 3 has 2 words
    lengths = torch.tensor(
        [len(question) for question in questions],
        dtype=torch.long
    )


    # Find the longest question in this particular batch.
    # Important:
    # We don't need every question in the entire dataset
    # to have the same length.
    # We only need the questions inside one batch to have
    # the same length.
    max_length = lengths.max().item()


    # Create a tensor filled with zeros.
    # 0 represents <PAD>.
    # Shape:
    # [batch_size, max_sequence_length]
    padded_questions = torch.zeros(
        len(questions),
        max_length,
        dtype=torch.long
    )


    # Put each actual question into the padded tensor.
    # Example:
    # Original:
    # [12, 43, 71]
    # becomes:
    # [12, 43, 71, 0, 0]
    # if max_length is 5.
    for i, question in enumerate(questions):
        padded_questions[
            i,
            :len(question)
        ] = question

    # Convert intents and categories into tensors.
    # torch.stack() combines multiple individual tensors
    # into one tensor.
    intents = torch.stack(intents)
    categories = torch.stack(categories)


    # Return everything required by the training loop.
    return (
        padded_questions,
        lengths,
        intents,
        categories
    )

# 14. CREATE DATASET OBJECTS

# Create the training Dataset.
train_dataset = FootballDataset(train_df)

# Create the testing Dataset.
test_dataset = FootballDataset(test_df)


# 15. CREATE DATALOADERS

# DataLoader automatically creates batches from our Dataset.
train_loader = DataLoader(
    train_dataset,
    # Number of questions processed together.
    batch_size=BATCH_SIZE,

    # Shuffle training examples after every epoch.
    # This helps prevent the model from learning based on
    # the original order of the dataset.
    shuffle=True,

    # Use our custom function for padding variable-length
    # questions.
    collate_fn=collate_fn
)


# Test DataLoader.
test_loader = DataLoader(
    test_dataset,

    # Same batch size.
    batch_size=BATCH_SIZE,

    # We don't need to shuffle test data.
    shuffle=False,

    # Use the same padding logic.
    collate_fn=collate_fn
)


# 16. RNN MODEL

class FootballRNN(nn.Module):
    """
    Neural network for football question classification.
    Architecture:
        Word IDs
           ↓
        Embedding
           ↓
        2-layer RNN
           ↓
        Final hidden state
           ↓
        Linear
           ↓
        ReLU
           ↓
        ┌───────────────┐
        ↓               ↓
      Intent         Category
       output          output
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_layers,
        num_intents,
        num_categories
    ):
        """
        Constructor for the RNN model.
        """

        # Initialize the parent nn.Module.
        super().__init__()


        # EMBEDDING LAYER

        # Convert each word ID into a learnable vector.
        # Example:
        # word ID 53
        # becomes:
        # [0.12, -0.45, 0.71, ..., 0.18]
        # with exactly 50 values.
        self.embedding = nn.Embedding(

            # Number of unique words in our vocabulary.
            vocab_size,

            # Number of dimensions in every word vector.
            embedding_dim,

            # 0 represents <PAD>.
            # PyTorch will treat the padding embedding
            # specially so that it does not get updated
            # like normal word embeddings.
            padding_idx=0
        )


        # RNN LAYER

        self.rnn = nn.RNN(

            # Size of each input word vector.
            # We chose 50.
            input_size=embedding_dim,

            # Number of hidden values maintained by
            # the RNN.
            # We chose 128.
            hidden_size=hidden_size,


            # Number of stacked RNN layers.
            # We chose 2.
            num_layers=num_layers,


            # Input/output tensor format:
            # [batch, sequence, features]
            # instead of:
            # [sequence, batch, features]
            batch_first=True
        )


        # FULLY CONNECTED LAYER

        # Convert the 128-dimensional final hidden
        # representation into 64 features.
        self.fc1 = nn.Linear(
            hidden_size,
            64
        )


        # ReLU introduces non-linearity.
        self.relu = nn.ReLU()


        # INTENT OUTPUT

        # Predict which intent the question belongs to.
        # If there are 100 unique intents:
        # output shape = [batch_size, 100]
        self.intent_output = nn.Linear(
            64,
            num_intents
        )


        # CATEGORY OUTPUT

        # Predict which football category the question
        # belongs to.
        # If there are 10 categories:
        # output shape = [batch_size, 10]
        self.category_output = nn.Linear(
            64,
            num_categories
        )


    def forward(self, x, lengths):
        """
        Defines how data moves through the network.
        x:
            padded word-index tensor
        lengths:
            actual length of every question
        """


        # STEP 1: WORD IDs → WORD EMBEDDINGS
        # Before embedding:
        # x shape:
        # [batch_size, sequence_length]
        # Example:
        # [32, 15]
        # After embedding:
        # [32, 15, 50]
        # because every word becomes a 50-dimensional vector.
        x = self.embedding(x)

        # STEP 2: PACK PADDED SEQUENCES
        # Our questions have different lengths.
        # We don't want the RNN to waste computation processing
        # <PAD> tokens.
        # pack_padded_sequence tells PyTorch the actual length
        # of every sequence.
        packed = nn.utils.rnn.pack_padded_sequence(

            # Embedded questions.
            x,

            # Actual question lengths.
            # lengths.cpu() ensures the lengths tensor is on
            # the CPU because this PyTorch utility expects it
            # there in this configuration.
            lengths.cpu(),

            # Our tensor format is:
            # [batch, sequence, features]
            batch_first=True,

            # We don't need to sort the questions from
            # longest to shortest manually.
            enforce_sorted=False
        )


        # STEP 3: RUN THE RNN

        # The RNN returns:
        # output
        # hidden
        # We only need the final hidden states for our
        # classification task.
        # Therefore "_" means:
        # "I intentionally don't need this value."
        _, hidden = self.rnn(packed)


        # STEP 4: SELECT FINAL RNN LAYER

        # hidden has shape:
        # [num_layers, batch_size, hidden_size]
        # With our configuration:
        # [2, batch_size, 128]
        # hidden[-1] selects the final RNN layer
        # Result:
        # [batch_size, 128]
        hidden = hidden[-1]


        # STEP 5: FULLY CONNECTED LAYER

        # Convert:
        # [batch_size, 128]
        # into:
        # [batch_size, 64]
        x = self.fc1(hidden)


        # Apply ReLU.
        x = self.relu(x)

        # STEP 6: TWO OUTPUT HEADS

        # Predict intent.
        intent = self.intent_output(x)


        # Predict category.
        category = self.category_output(x)


        # Return both predictions.
        return intent, category


# 17. SELECT DEVICE

# If CUDA is available:
#
#     use NVIDIA GPU
#
# Otherwise:
#
#     use CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# Display which device is being used.
print("Device:", device)


# Display useful information about our dataset.
print("Vocabulary size:", len(word_to_index))
print("Intent classes:", len(intent_to_index))
print("Categories:", len(category_to_index))
print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))


# 18. CREATE MODEL

model = FootballRNN(

    # Number of unique words.
    vocab_size=len(word_to_index),

    # 50-dimensional word vectors.
    embedding_dim=EMBEDDING_DIM,

    # 128 hidden-state dimensions.
    hidden_size=HIDDEN_SIZE,

    # 2 stacked RNN layers.
    num_layers=NUM_LAYERS,

    # Number of intent classes.
    num_intents=len(intent_to_index),

    # Number of category classes.
    num_categories=len(category_to_index)

).to(device)


# 19. LOSS FUNCTION

# CrossEntropyLoss is appropriate because both our
# predictions are classification problems.
# Intent:
#     question → intent class
# Category:
#     question → category class
criterion = nn.CrossEntropyLoss()


# 20. OPTIMIZER-

# Adam updates all trainable parameters of the model.
# This includes:
# embedding weights
# RNN weights
# RNN biases
# Linear layer weights
# Linear layer biases
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# 21. TRAINING LOOP

# This loop runs once for every epoch.
# If EPOCHS = 25:
# this loop runs 25 times.
for epoch in range(EPOCHS):


    # Put the model into training mode.
    model.train()


    # Variables used to calculate statistics for this epoch.
    total_loss = 0
    correct_intents = 0
    correct_categories = 0
    total = 0


    # BATCH LOOP
    # This loop processes one batch at a time.
    # If we have 800 training samples and BATCH_SIZE = 32,
    # there will be approximately:
    # 800 / 32 = 25 batches
    # per epoch.
    for (
        questions,
        lengths,
        intents,
        categories
    ) in train_loader:


        # Move question tensors to the selected device.
        questions = questions.to(device)


        # Move sequence lengths to the selected device.
        lengths = lengths.to(device)


        # Move intent labels to the selected device.
        intents = intents.to(device)


        # Move category labels to the selected device.
        categories = categories.to(device)

        # CLEAR OLD GRADIENTS
        # PyTorch accumulates gradients by default.
        # We therefore clear the gradients before processing
        # the current batch.
        optimizer.zero_grad()


        # FORWARD PASS
        # Send the batch through our RNN.
        # The model returns:
        # intent_output
        # category_output
        intent_output, category_output = model(
            questions,
            lengths
        )


        # INTENT LOSS
        # Compare the predicted intent with the actual intent.
        intent_loss = criterion(
            intent_output,
            intents
        )


        # CATEGORY LOSS
        # Compare the predicted category with the actual
        # category.
        category_loss = criterion(
            category_output,
            categories
        )


        # COMBINED LOSS
        # Intent is our main objective.
        # Category is an auxiliary learning objective.
        # Therefore:
        # total loss =
        # intent loss
        # +
        # 0.3 × category loss
        # The 0.3 means category contributes less strongly
        # than intent.
        loss = (
            intent_loss
            +
            0.3 * category_loss
        )


        # BACKPROPAGATION
        # Calculate gradients of all trainable parameters.
        # The gradient flows backward through:
        # output layers
        # ↓
        # Linear
        # ↓
        # RNN
        # ↓
        # Embedding
        loss.backward()


        # UPDATE PARAMETERS
        # Adam uses the gradients calculated above to
        # update the model parameters.
        optimizer.step()


        # LOSS STATISTICS
        # Add this batch's loss to the total epoch loss.
        total_loss += loss.item()


        # INTENT PREDICTIONS
        # Find the class with the highest logit.
        # dim=1 means:
        # select the largest value across the class dimension.
        intent_predictions = torch.argmax(
            intent_output,
            dim=1
        )

        # CATEGORY PREDICTIONS
        category_predictions = torch.argmax(
            category_output,
            dim=1
        )


        # COUNT CORRECT INTENT PREDICTIONS

        # Compare predicted intent with actual intent.
        # Example:
        # predictions:
        # [1, 4, 3, 2]
        # actual:
        # [1, 2, 3, 2]
        # correct:
        # [True, False, True, True]
        # sum() = 3
        correct_intents += (
            intent_predictions == intents
        ).sum().item()


        # Count correct category predictions.
        correct_categories += (
            category_predictions == categories
        ).sum().item()


        # Add number of examples in this batch.
        total += intents.size(0)


    # CALCULATE TRAINING METRICS
    # Average loss across all batches.
    train_loss = total_loss / len(train_loader)


    # Training intent accuracy.
    intent_accuracy = (
        correct_intents / total
    )


    # Training category accuracy.
    category_accuracy = (
        correct_categories / total
    )

    # TESTING
    # Switch model to evaluation mode.
    model.eval()


    # Number of correct test predictions.
    test_correct = 0


    # Total number of test examples.
    test_total = 0


    # Disable gradient calculations.
    # We aren't training here.
    # This saves memory and computation.
    with torch.no_grad():


        # Process every test batch.
        for (
            questions,
            lengths,
            intents,
            categories
        ) in test_loader:


            # Move inputs to device.
            questions = questions.to(device)
            lengths = lengths.to(device)
            intents = intents.to(device)


            # We only care about intent during testing.
            # Therefore category output is ignored.
            intent_output, _ = model(
                questions,
                lengths
            )


            # Get predicted intent class.
            predictions = torch.argmax(
                intent_output,
                dim=1
            )


            # Count correct predictions.
            test_correct += (
                predictions == intents
            ).sum().item()


            # Count total examples.
            test_total += intents.size(0)


    # Calculate test accuracy.
    test_accuracy = (
        test_correct / test_total
    )

    # DISPLAY EPOCH RESULTS

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Loss: {train_loss:.4f} "
        f"Intent Acc: {intent_accuracy:.4f} "
        f"Category Acc: {category_accuracy:.4f} "
        f"Test Acc: {test_accuracy:.4f}"
    )


# 22. BUILD ANSWER MAP

# We want to convert:
# predicted intent
#       ↓
# original intent ID
#       ↓
# answer
# Example:
# 198
# ↓
# "Claudio Ranieri was the manager..."
answer_map = {}


# Go through every row in the original dataset.
for _, row in df.iterrows():

    # Store the answer using intent_id as the key.
    #
    # Example:
    #
    # answer_map[198] =
    # "Claudio Ranieri was the manager..."
    answer_map[
        int(row["intent_id"])
    ] = row["answer"]


# ------------------------------------------------------------
# 23. CREATE MODEL CHECKPOINT
# ------------------------------------------------------------

# We don't only save the neural network weights.
#
# We also save everything required to process new questions.
checkpoint = {

    # Learned neural network parameters.
    "model_state": model.state_dict(),


    # Word → integer mapping.
    # Required when converting a new question into
    # numerical input.
    "vocab": word_to_index,


    # Original intent ID → model class index.
    "intent_to_index": intent_to_index,


    # Model class index → original intent ID.
    "index_to_intent": index_to_intent,


    # Category → model class index.
    "category_to_index": category_to_index,


    # Model class index → category.
    "index_to_category": index_to_category,


    # Intent → answer.
    "answer_map": answer_map,


    # Save these hyperparameters because the exact same
    # architecture is required when loading the model.
    "embedding_dim": EMBEDDING_DIM,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS
}


# 24. SAVE MODEL

# Save the entire checkpoint into:
# rnn1.pth
# This file will contain:
# model weights
# vocabulary
# intent mappings
# category mappings
# answers
# architecture configuration
torch.save(
    checkpoint,
    "rnn1.pth"
)


# Tell us that saving was successful.
print("Model saved as rnn1.pth")