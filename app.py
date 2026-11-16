import os, re
from urllib.parse import urlparse, parse_qs
import joblib, pandas as pd, plotly.express as px, requests, streamlit as st, nltk
from nltk.corpus import stopwords

st.set_page_config(page_title="SentimentAI", page_icon="💬", layout="wide")
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 15% 0%,#211447 0,#100d20 38%,#0b0a14 100%)}
section[data-testid="stSidebar"]{background:#100d20;border-right:1px solid #39265f}
.block-container{padding-top:2rem}
h1,h2,h3{color:#f7f3ff!important}
div[data-testid="stMetric"]{background:linear-gradient(135deg,#21183b,#171326);border:1px solid #52368a;padding:16px;border-radius:14px}
div.stButton>button[kind="primary"]{background:linear-gradient(90deg,#7c3aed,#a855f7);border:0;border-radius:10px}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    if not (os.path.exists("sentiment_model.pkl") and os.path.exists("tfidf_vectorizer.pkl")):
        return None, None
    return joblib.load("sentiment_model.pkl"), joblib.load("tfidf_vectorizer.pkl")

@st.cache_resource
def get_stop_words():
    try:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english")) - {"not","no","nor","never"}
    except Exception:
        return set()

model, vectorizer = load_models()
stop_words = get_stop_words()
LABELS = {-1:"Negative", 0:"Neutral", 1:"Positive"}
COLORS = {"Positive":"#34d399","Neutral":"#fbbf24","Negative":"#fb7185"}

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\S+|@\\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return " ".join(w for w in text.split() if w not in stop_words) if stop_words else text

def show_prediction(text, source):
    if model is None or vectorizer is None:
        st.error("Missing sentiment_model.pkl or tfidf_vectorizer.pkl beside app.py.")
        return
    cleaned = preprocess_text(text)
    if not cleaned:
        st.warning("No usable English text remained after preprocessing.")
        return
    vector = vectorizer.transform([cleaned])
    pred = int(model.predict(vector)[0])
    st.subheader("Analysis result")
    st.success(f"Predicted sentiment: {LABELS[pred]}")
    st.caption(source)
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(vector)[0]
        df = pd.DataFrame({"Sentiment":[LABELS[int(c)] for c in model.classes_],
                           "Estimated probability (%)":[float(p)*100 for p in raw]})
        fig = px.bar(df, x="Sentiment", y="Estimated probability (%)", color="Sentiment",
                     color_discrete_map=COLORS, title="Model-estimated probabilities")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", yaxis_range=[0,100], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("View processed text"):
        st.write(cleaned)

@st.cache_data
def dataset_charts():
    frames=[]
    if os.path.exists("Twitter_Data.csv"):
        d=pd.read_csv("Twitter_Data.csv",usecols=["category"]); d["Platform"]="Twitter / X"; frames.append(d)
    if os.path.exists("Reddit_Data.csv"):
        d=pd.read_csv("Reddit_Data.csv",usecols=["category"]); d["Platform"]="Reddit"; frames.append(d)
    if not frames: return None,None
    data=pd.concat(frames,ignore_index=True).dropna(subset=["category"])
    data["Sentiment"]=data["category"].astype(int).map(LABELS)
    data=data.dropna(subset=["Sentiment"])
    summary=data.groupby("Sentiment").size().reindex(["Positive","Neutral","Negative"],fill_value=0).reset_index(name="Posts")
    by_platform=data.groupby(["Platform","Sentiment"]).size().reset_index(name="Posts")
    return summary,by_platform

def platform(url):
    host=urlparse(url).netloc.lower().replace("www.","")
    if "reddit.com" in host or "redd.it" in host: return "Reddit"
    if host in ("x.com","twitter.com","mobile.twitter.com"): return "X / Twitter"
    if "instagram.com" in host: return "Instagram"
    if "youtube.com" in host or host=="youtu.be": return "YouTube"
    return "Unknown"

def reddit_text(url):
    response=requests.get(url.split("?")[0].rstrip("/")+".json",
        headers={"User-Agent":"SentimentAI academic project/1.0"},timeout=20)
    response.raise_for_status(); data=response.json()
    post=data[0]["data"]["children"][0]["data"]
    parts=[post.get("title",""),post.get("selftext","")]
    for item in data[1]["data"]["children"]:
        body=item.get("data",{}).get("body","")
        if body and body not in ("[deleted]","[removed]"): parts.append(body)
    return "\\n".join(x for x in parts if x.strip()),len(parts)

def youtube_text(url,key):
    parsed=urlparse(url); vid=(parsed.path.strip("/") if parsed.netloc.endswith("youtu.be") else
        parse_qs(parsed.query).get("v",[None])[0] if parsed.path=="/watch" else
        parsed.path.split("/")[2] if parsed.path.startswith(("/shorts/","/embed/")) else None)
    if not vid: raise ValueError("Could not identify the YouTube video ID.")
    r=requests.get("https://www.googleapis.com/youtube/v3/commentThreads",
        params={"part":"snippet","videoId":vid,"maxResults":100,"textFormat":"plainText","key":key},timeout=20)
    r.raise_for_status()
    comments=[x["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for x in r.json().get("items",[])]
    if not comments: raise ValueError("No public comments found; comments may be disabled.")
    return "\\n".join(comments),len(comments)

def x_text(url,token):
    match=re.search(r"/status/(\\d+)",url)
    if not match: raise ValueError("X URL must contain /status/POST_ID.")
    r=requests.get("https://api.x.com/2/tweets/"+match.group(1),
        headers={"Authorization":"Bearer "+token},params={"tweet.fields":"text"},timeout=20)
    r.raise_for_status()
    text=r.json().get("data",{}).get("text","")
    if not text: raise ValueError("X returned no post text.")
    return text,1

st.sidebar.markdown("# 💬 SentimentAI")
st.sidebar.caption("Social Media Sentiment Analyzer")
page=st.sidebar.radio("Navigation",["Dashboard","Analyze Post (URL)","Analyze Text","Model Performance","About Project"])
st.sidebar.markdown("---")
st.sidebar.caption("Reddit · X · Instagram · YouTube")
if model is None or vectorizer is None:
    st.error("Put sentiment_model.pkl and tfidf_vectorizer.pkl in the same folder as app.py.")

if page=="Dashboard":
    st.title("📊 SentimentAI Dashboard")
    st.caption("Social media sentiment analysis using your trained machine-learning model.")
    summary,platforms=dataset_charts()
    if summary is not None:
        total=int(summary.Posts.sum()); counts=dict(zip(summary.Sentiment,summary.Posts))
        a,b,c,d=st.columns(4)
        a.metric("Dataset posts",f"{total:,}")
        b.metric("Positive",f"{counts['Positive']/total:.1%}")
        c.metric("Neutral",f"{counts['Neutral']/total:.1%}")
        d.metric("Negative",f"{counts['Negative']/total:.1%}")
        left,right=st.columns(2)
        with left:
            fig=px.pie(summary,names="Sentiment",values="Posts",hole=.58,color="Sentiment",
                color_discrete_map=COLORS,title="Dataset sentiment distribution")
            fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
        with right:
            fig=px.bar(platforms,x="Platform",y="Posts",color="Sentiment",barmode="group",
                color_discrete_map=COLORS,title="Sentiment by source platform")
            fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
    else:
        st.info("Add Reddit_Data.csv and Twitter_Data.csv beside app.py to display dataset charts.")
    st.write("Use Analyze Post (URL) or Analyze Text to run predictions.")

elif page=="Analyze Post (URL)":
    st.title("🔗 Analyze a Social Media Post")
    st.caption("Automatic extraction depends on public access and platform API permissions.")
    url=st.text_input("Post URL",placeholder="Paste a Reddit, X, Instagram or YouTube URL")
    detected=platform(url) if url.strip() else "Not detected"
    st.info("Detected platform: "+detected)
    pasted=st.text_area("Caption / post text / comments (fallback)",height=130,
        placeholder="Paste the caption or comments if automatic extraction is unavailable.")
    with st.expander("API credentials"):
        st.write("Add keys in Streamlit Community Cloud → App settings → Secrets. Do not hardcode keys in public code.")
        st.code('YOUTUBE_API_KEY = "your_key"\\nX_BEARER_TOKEN = "your_token"',language="toml")
        st.caption("Reddit public JSON may work for public posts. YouTube and X need API credentials. Instagram URL-only extraction is not reliably available here; paste its caption/comments.")
    if st.button("Analyze Post",type="primary"):
        content=pasted.strip()
        try:
            if not content and detected=="Reddit":
                with st.spinner("Fetching Reddit post and comments..."): content,n=reddit_text(url)
                st.success(f"Retrieved text from about {n} post/comment entries.")
            elif not content and detected=="YouTube":
                key=st.secrets.get("YOUTUBE_API_KEY",os.getenv("YOUTUBE_API_KEY",""))
                if not key: raise ValueError("Add YOUTUBE_API_KEY to Streamlit Secrets or paste comments above.")
                with st.spinner("Fetching YouTube comments..."): content,n=youtube_text(url,key)
                st.success(f"Retrieved {n} comments.")
            elif not content and detected=="X / Twitter":
                token=st.secrets.get("X_BEARER_TOKEN",os.getenv("X_BEARER_TOKEN",""))
                if not token: raise ValueError("Add X_BEARER_TOKEN to Streamlit Secrets or paste post text above.")
                with st.spinner("Fetching X post..."): content,n=x_text(url,token)
                st.success("Retrieved X post.")
            elif not content and detected=="Instagram":
                raise ValueError("Paste the Instagram caption/comments. Official API access requires eligible Meta permissions and media access.")
            elif not content and detected=="Unknown":
                raise ValueError("Unsupported URL. Use Reddit, X, Instagram or YouTube.")
            if content: show_prediction(content,"Source: "+detected)
        except requests.RequestException as e:
            st.error(f"Could not retrieve the post: {e}. Check URL, permissions and visibility.")
        except (ValueError,KeyError,IndexError) as e:
            st.warning(str(e))

elif page=="Analyze Text":
    st.title("✍️ Analyze Text")
    user_text=st.text_area("Paste a tweet, comment, caption or post",height=180)
    if st.button("Analyze Sentiment",type="primary"):
        if not user_text.strip(): st.warning("Please enter some text.")
        else: show_prediction(user_text,"Manually entered text")

elif page=="Model Performance":
    st.title("🏆 Model Performance")
    if model is not None:
        st.metric("Selected trained model",type(model).__name__)
        st.write("The saved trained model is loaded successfully.")
        st.info("This page does not invent accuracy scores or a confusion matrix. Add the measured evaluation results from your notebook to display them.")
        st.write("Model classes:",[LABELS.get(int(x),str(x)) for x in model.classes_])

else:
    st.title("ℹ️ About SentimentAI")
    st.write("SentimentAI classifies social-media text as Positive, Neutral or Negative.")
    st.markdown("**Workflow**\\n1. Twitter and Reddit dataset collection\\n2. Data cleaning and preprocessing\\n3. TF-IDF feature extraction\\n4. Naive Bayes, Logistic Regression and Decision Tree comparison\\n5. Model evaluation and prediction")
    st.warning("The saved model was trained on the supplied Twitter and Reddit dataset. Instagram and YouTube text is analyzed by the same model, not by platform-specific models. Predictions may be less reliable for unfamiliar topics or languages.")
    st.caption("Built with Python, Streamlit, scikit-learn, NLTK, Pandas and Plotly.")
