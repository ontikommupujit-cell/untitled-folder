import requests
from config import HF_API_KEY
MODEL_ID = "facebook/bart-large-mnli"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
HEADERS= {"Authorization": f"Bearer {HF_API_KEY}"}
TOPICS=["sports","Technology","Politics","Health","Business"]
def ask_hf(headline:str):
    payload={"inputs":headline,"parameters":{"candidate_labels":TOPICS}}
    r=requests.post(API_URL,headers=HEADERS, json=payload,timeout=30)
    if not r.ok:
        raise RuntimeError(f"API error: {r.status_code} {r.text}")
    return r.json()
def best_topic(preds:list):
    best=max(preds,key=lambda x:x[1])
    return best["label"],best["score"]
def bar(score:float)-> str:
    pct=score*100
    blocks=int(pct//10)
    return "🟩" * blocks+"⬜"*(10-blocks)
def show(headline:str,preds:list):
    top_label,top_score=best_topic(preds)
    print("\n"+"="*60)
    print('???? News Topic Classifier')
    print("="*60)
    print(f"Headline: {headline}")
    print(f'Best Topic:{top_label}')
    print(f"Confidence:{round(top_score*100,2)}% {bar(top_score)}")
    print('\nTop 3 guesses')
    top3=sorted(preds,key=lambda x:x[1],reverse=True)[:3]
    for i, p in enumerate(top3,1):
        print(f"{i}], {p['label']}: {round(p['score']*100,2)}% {bar(p['score'])}")
        print("="*60)
def main():
    print('Welcome to the News Topic Classifier')
    print('Type a news headline and ill try to guess its topic.')
    print("Type 'exit' to quit.\n")
    while True:
        headline=input('Headline:').strip()
        if headline.lower()=='exit':
            print('Goodbye!')
            break
        if not headline:
            print('Please enter a valid headline.')
            continue
        try:
            preds = ask_hf(headline)
            if isinstance(preds, list) and preds and "label" in preds[0]:
                show(headline, preds)
            else:
                print("Oops! Unexpected reply:", preds)
        except Exception as e:
            print("\n⚠️ Oops! Something went wrong.")
            print("Reason:", e)
            print("Tip: Check HF_API_KEY + internet.\n")
if __name__ == "__main__":

     main() 