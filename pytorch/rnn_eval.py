
import re
import torch
import torch.nn as nn


# CONFIGURATION
MODEL_PATH = "rnn1.pth"

CONFIDENCE_THRESHOLD = 0.50

# TOKENIZATION


def tokenize(text):
    """
    Converts a sentence into individual words.

    Example:

    "Who won the 1994 World Cup?"

    becomes:

    ["who", "won", "the", "1994", "world", "cup"]
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    return text.split()


# RNN MODEL

class FootballRNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_layers,
        num_intents,
        num_categories
    ):
        super().__init__()

        # Converts each word index into an embedding vector.
        # Example:
        # word index = 25
        # becomes:
        # [0.12, -0.45, 0.31, ...]
        # with 50 values.

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # RNN processes the sequence of word embeddings.

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Classification layer.

        self.fc1 = nn.Linear(
            hidden_size,
            64
        )

        self.relu = nn.ReLU()

        # Predicts the intent.

        self.intent_output = nn.Linear(
            64,
            num_intents
        )

        # Predicts the category.

        self.category_output = nn.Linear(
            64,
            num_categories
        )


    def forward(self, x, lengths):

        # Convert word indices into embedding vectors

        x = self.embedding(x)

        # Remove padding from the actual RNN computation
        # pack_padded_sequence tells the RNN the real
        # length of each question.

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Run the sequence through the RNN
        _, hidden = self.rnn(packed)

        # hidden shape:
        # [number_of_layers, batch_size, hidden_size]
        # We only have one question during evaluation,
        # so:
        # [2, 1, 128]
        # hidden[-1] selects the final RNN layer.

        hidden = hidden[-1]

        # Fully connected layer

        x = self.fc1(hidden)

        x = self.relu(x)

        # Produce predictions

        intent = self.intent_output(x)

        category = self.category_output(x)

        return intent, category


# LOAD MODEL CHECKPOINT

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# LOAD INFORMATION STORED DURING TRAINING

word_to_index = checkpoint["vocab"]

intent_to_index = checkpoint["intent_to_index"]

index_to_intent = checkpoint["index_to_intent"]

category_to_index = checkpoint["category_to_index"]

index_to_category = checkpoint["index_to_category"]

answer_map = checkpoint["answer_map"]


embedding_dim = checkpoint["embedding_dim"]

hidden_size = checkpoint["hidden_size"]

num_layers = checkpoint["num_layers"]


# ============================================================
# CREATE MODEL
# ============================================================

model = FootballRNN(
    vocab_size=len(word_to_index),
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_intents=len(intent_to_index),
    num_categories=len(category_to_index)
)


# Load the learned weights.

model.load_state_dict(
    checkpoint["model_state"]
)


# Move model to CPU/GPU.

model = model.to(device)


# Evaluation mode disables training-specific behavior.

model.eval()


# ============================================================
# ENCODE USER QUESTION
# ============================================================

def encode_question(question):

    tokens = tokenize(question)

    # If the user enters an empty question.

    if len(tokens) == 0:
        return None

    encoded = []

    for word in tokens:

        # If the word exists in the vocabulary,
        # use its index.
        #
        # Otherwise use <UNK>.

        index = word_to_index.get(
            word,
            word_to_index["<UNK>"]
        )

        encoded.append(index)

    return encoded


# PREDICT ANSWER

def predict_answer(question):

    encoded_question = encode_question(
        question
    )

    if encoded_question is None:
        return None


    # --------------------------------------------------------
    # Convert question into PyTorch tensor
    # Current shape:
    # [sequence_length]
    # Example:
    # [12, 45, 87, 23, 91]
    question_tensor = torch.tensor(
        encoded_question,
        dtype=torch.long
    )


    # Add batch dimension
    # Before:
    # [sequence_length]
    # After:
    # [1, sequence_length]
    # 1 = one question

    question_tensor = question_tensor.unsqueeze(0)

    # Store the real sequence length

    lengths = torch.tensor(
        [len(encoded_question)],
        dtype=torch.long
    )


    # Move tensors to same device as model.

    question_tensor = question_tensor.to(device)

    lengths = lengths.to(device)

    # Run model

    with torch.no_grad():

        intent_output, category_output = model(
            question_tensor,
            lengths
        )

    # Convert intent logits into probabilities

    intent_probabilities = torch.softmax(
        intent_output,
        dim=1
    )
    
    # Find the highest probability

    confidence, predicted_index = torch.max(
        intent_probabilities,
        dim=1
    )


    confidence = confidence.item()

    predicted_index = predicted_index.item()


    # Convert predicted index back into intent_id

    predicted_intent_id = index_to_intent[
        predicted_index
    ]

    # Check confidence

    if confidence < CONFIDENCE_THRESHOLD:

        return {
            "answer": "I'm not confident enough to answer that question.",
            "intent_id": predicted_intent_id,
            "confidence": confidence,
            "known": False
        }


    # Find answer associated with predicted intent

    answer = answer_map.get(
        int(predicted_intent_id)
    )


    # If no answer exists for the predicted intent

    if answer is None:

        return {
            "answer": "I don't have an answer for that question.",
            "intent_id": predicted_intent_id,
            "confidence": confidence,
            "known": False
        }


    return {
        "answer": answer,
        "intent_id": predicted_intent_id,
        "confidence": confidence,
        "known": True
    }


# INTERACTIVE CHAT LOOP

print()
print("==============================================")
print("        FOOTBALL RNN QUESTION ANSWER")
print("==============================================")
print()
print("Model loaded successfully.")
print("Type 'exit' to stop.")
print()


while True:

    question = input("You: ").strip()


    # Exit condition

    if question.lower() in [
        "exit",
        "quit",
        "q"
    ]:

        print("Goodbye!")

        break


    # Empty input

    if not question:

        print("Please enter a question.")

        continue

    # Get prediction

    result = predict_answer(
        question
    )

    # Display answer

    print()
    print("RNN:", result["answer"])

    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Predicted Intent: "
        f"{result['intent_id']}"
    )

    print()

