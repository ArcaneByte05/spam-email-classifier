"""Run with: python -m streamlit run app.py"""
import streamlit as st

from train import DATA_FILE, clean_email, train_model

st.title("Spam email classifier")
st.write("Paste an English email to check whether the model considers it spam.")

if not DATA_FILE.exists():
    st.error("Download enron_spam_data.zip into the project folder first. "
             "See step 5 in README.md, then refresh this page.")
    st.stop()


@st.cache_resource
def get_model():
    # Hold the trained model in memory so each click is quick.
    return train_model()


with st.spinner("Training on the labelled emails..."):
    model, results = get_model()

email = st.text_area("Email subject and body", placeholder="Team meeting tomorrow at 10...")
if st.button("Check email"):
    text = clean_email(email)
    if not text:
        st.warning("Please enter an email with some words.")
    elif model.named_steps["tfidf"].transform([text]).nnz == 0:
        st.warning("The model has never seen these words. Try adding more email content.")
    else:
        label = model.predict([text])[0]
        st.success("Prediction: spam" if label == "spam" else "Prediction: not spam (ham)")

st.subheader("Test results")
st.metric("Held-out accuracy", f"{results['accuracy']:.2%}")
st.caption(f"{results['correct']:,} correct out of {results['test_size']:,} test emails. "
           f"Trained on {results['train_size']:,} other emails.")
st.subheader("Email distribution")
st.caption("Number of emails in each class after cleaning, before the split.")
counts = results["counts"].rename(index={"ham": "not spam (ham)"})
st.bar_chart(counts.rename("Emails"))
