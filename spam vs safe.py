
import requests
from config import HF_API_KEY

MODEL = "facebook/bart-large-mnli"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
LABELS = ["Spam", "Safe"]

def classify_message(message):
    payload = {
        "inputs": message,
        "parameters": {
            "candidate_labels": LABELS
        }
    }

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        timeout=30
    )

    if not response.ok:
        raise RuntimeError(
            f"API error: {response.status_code} {response.text}"
        )

    data = response.json()

    if "labels" not in data or "scores" not in data:
        raise ValueError("Unexpected API response.")

    results = list(zip(data["labels"], data["scores"]))
    return sorted(results, key=lambda x: x[1], reverse=True)

def show_results(message, results):
    label, score = results[0]

    print("\n" + "=" * 44)
    print("Spam vs Safe Message Classifier")
    print("=" * 44)
    print(f"Message: {message}")
    print(f"Result: {label} ({score * 100:.1f}%)")
    print("\nConfidence Scores:")

    for i, (lbl, scr) in enumerate(results, 1):
        print(f"{i}. {lbl}: {scr * 100:.1f}%")

    if label == "Spam":
        print("\nWarning: Avoid suspicious links and requests.")
    else:
        print("\nLooks safe, but stay alert.")

    print("=" * 44)

def main():
    print("Spam vs Safe Message Classifier")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            msg = input("Enter a message: ").strip()

            if msg.lower() == "exit":
                print("Goodbye!")
                break

            if not msg:
                print("Please enter a message.\n")
                continue

            results = classify_message(msg)
            show_results(msg, results)

        except requests.exceptions.Timeout:
            print("\nError: Request timed out. Try again.\n")

        except requests.exceptions.RequestException as e:
            print(f"\nNetwork error: {e}\n")

        except (RuntimeError, ValueError, KeyError) as e:
            print(f"\nError: {e}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
