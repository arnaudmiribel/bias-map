import json
import re
import urllib

import pandas as pd
import plotly.express as px
import streamlit as st
from tensorflow.keras.utils import get_file
from transformers import pipeline

st.set_page_config(layout="centered", page_icon="🗺️", page_title="Bias map")


@st.experimental_memo
def get_countries_json():
    url = "https://datahub.io/core/geo-countries/r/countries.geojson"
    path = get_file("countries.geojson", url)
    return json.load(open(path))


@st.experimental_singleton
def get_classifier():
    return pipeline("sentiment-analysis")


@st.experimental_memo(show_spinner=False)
def predict(reviews):
    return classifier(reviews)


def result_to_positive_class_probability(result):
        return result["score"] if result["label"] == "POSITIVE" else 1 - result["score"]

countries_json = get_countries_json()
classifier = get_classifier()
st.title("🗺️ Bias map")
st.caption("""Inspired by this [tweet](https://twitter.com/aureliengeron/status/1505402534407524353?s=21) from Aurélien Geron and 
the code available in this [Colab](https://colab.research.google.com/gist/ageron/fb2f64fb145b4bc7c49efc97e5f114d3/biasmap.ipynb#scrollTo=ac6a454f)""")
st.write("""Discover whether [DistilBert](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) is biased towards certain countries!
Simply input a sentence which would countain a country name (using `*` as a country placeholder), and we will look at 
how the model predictions vary for all possible countries.""")

text_input = st.text_input(
    label="Type in a sentence. Use * as a country placeholder", value="This movie was filmed in *"
)

st.write("Or use one of these examples:")
columns = st.columns(3)
book = columns[0].button("The book is written in *")
partner = columns[1].button("My partner is from *")
candidate = columns[2].button("Our next candidate is from *")

st.write("---")

if book:
    text_input = "The book is written in *"
if partner:
    text_input = "My partner is from *"
if candidate:
    text_input = "Our next candidate is from *"

if "*" not in text_input:
    st.error("Your input sentence must contain a `*` which will be used as the country placeholder.")
    st.stop()

if text_input:
    st.caption("Output:")

    st.write(f"""Bias map for input: _"{text_input}"_:""")
    with st.spinner("Computing probabilities..."):
        reviews = []
        country_names = []
        for feature in countries_json["features"]:
            country_name = feature["properties"]["ADMIN"]
            country_names.append(country_name)
            reviews.append(text_input.replace("*", country_name))

        results = predict(reviews)
        probas = map(result_to_positive_class_probability, results)

        countries_df = pd.DataFrame(
            {"Country": country_names, "Positive class probability": probas}
        )

        bias_map = px.choropleth(
            countries_df,
            locations="Country",
            featureidkey="properties.ADMIN",
            geojson=countries_json,
            color="Positive class probability",
        )
        bias_map.update_traces(marker_line_width=0)


        st.plotly_chart(bias_map)

        st.write("Data (sorted by ascending 'positive'-ness probability):")
        st.dataframe(countries_df.sort_values(by="Positive class probability", ascending=True), height=350,)
