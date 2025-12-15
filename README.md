# SpiritedData

**Exploring the emotional heartbeat of Studio Ghibli films**

🌐 **Live Demo**: [spiriteddata.streamlit.app](https://spiriteddata.streamlit.app)

## About

SpiritedData is an emotion analysis engine that decodes the emotional DNA of Studio Ghibli's filmography using multilingual NLP and signal processing. Every line of dialogue across 22 films and 5 languages has been analyzed through a 28-dimension emotion classifier.

## Features

- 🎬 **The Spirit Archives** - Deep-dive film emotion analysis with interactive timelines
- 🌍 **Echoes Across Languages** - Cross-language emotion comparison (EN, FR, ES, NL, AR)
- 🎭 **Architects of Emotion** - Director emotional style profiles (Miyazaki vs Takahata)
- 📊 **The Alchemy of Data** - Methodology & data quality transparency
- 🧠 **Memories of Sora** - AI assistant retrospective and lessons learned

## Technology Stack

- **Frontend**: Streamlit
- **Database**: DuckDB (embedded analytics)
- **Visualization**: Plotly
- **NLP**: HuggingFace Transformers (GoEmotions model)

## Data

- 22 Studio Ghibli films analyzed
- 5 languages: English, French, Spanish, Dutch, Arabic
- 28 emotion dimensions per dialogue entry
- ~100,000 dialogue entries processed

## Local Development

```bash
pip install -r requirements.txt
streamlit run Home.py
```

## Credits

Built by Ed Junior as a data engineering portfolio project.

Analysis powered by the [multilingual_go_emotions](https://huggingface.co/AnasAlokla/multilingual_go_emotions) model.

