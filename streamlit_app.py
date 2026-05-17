import streamlit as st
from transformers import pipeline
import pandas as pd

# Configure the page layout and title
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

@st.cache_resource
def load_pipeline():
    # Cache the model so it doesn't reload on every interaction
    return pipeline("sentiment-analysis")

def main():
    st.title("Advanced Sentiment Analysis")
    st.markdown("Analyze the sentiment of your text instantly using Hugging Face Transformers.")

    # Initialize session state to keep track of history
    if "history" not in st.session_state:
        st.session_state.history = []

    sentiment_analyzer = load_pipeline()

    # Form layout for better user experience
    with st.form(key="analyze_form"):
        text = st.text_area(
            "Enter your text below:", 
            height=150, 
            placeholder="E.g., I absolutely loved the new movie! The acting was fantastic..."
        )
        submit_button = st.form_submit_button(label="Analyze Sentiment")

    if submit_button:
        if text.strip():
            with st.spinner("Analyzing sentiment..."):
                results = sentiment_analyzer(text)[0]
                label = results['label']
                score = results['score']

                # Save to history
                st.session_state.history.insert(0, {"Text": text, "Sentiment": label, "Confidence": f"{score:.2%}"})

                # Display Results beautifully
                st.markdown("### Results")
                
                # Determine colors and icons based on sentiment
                if label == "POSITIVE":
                    st.success(f"**Sentiment:** {label}")
                elif label == "NEGATIVE":
                    st.error(f"**Sentiment:** {label}")
                else:
                    st.info(f"**Sentiment:** {label}")

                # Show metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Detected Sentiment", value=label)
                with col2:
                    st.metric(label="Confidence Score", value=f"{score:.2%}")
                
                # Show a progress bar for visual indication of confidence
                st.progress(float(score), text="Confidence Level")

        else:
            st.warning("Please enter some text to analyze.")

    # Display History
    if st.session_state.history:
        st.markdown("---")
        st.subheader("Recent Analyses History")
        
        # Convert history to DataFrame for nice table display
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

if __name__ == "__main__":
    main()
