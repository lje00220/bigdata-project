import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="유튜브 댓글 분석", layout="wide")

plt.rc('font', family='AppleGothic')  
plt.rcParams['axes.unicode_minus'] = False  

@st.cache_data
def load_data():
    return pd.read_csv('유튜브 댓글 모음(세번째 질문 완료).csv')

data = load_data()

st.title("유튜브 댓글 분석 프로젝트")

# 1. AI 봇과 일반 사용자의 감정 점수 비교
st.header("1. AI 봇과 일반 사용자의 감정 점수 비교")

fig, ax = plt.subplots(figsize=(6, 4))

ai_bot_data = data[data['bot'] == 1]
user_data = data[data['bot'] == 0]

sns.histplot(ai_bot_data['sentiment_score'], bins=30, kde=True, color='skyblue', label='AI 봇', ax=ax, alpha=0.5)
sns.histplot(user_data['sentiment_score'], bins=30, kde=True, color='salmon', label='일반 사용자', ax=ax, alpha=0.5)

ax.set_title("AI 봇과 일반 사용자의 감정 점수 분포", fontsize=16)
ax.set_xlabel("감정 점수", fontsize=12)
ax.set_ylabel("빈도", fontsize=12)
ax.legend(title="유형", fontsize=10)

st.pyplot(fig)

# 2. 감정 비율 비교 (긍정/부정)
st.header("2. 감정 비율 분석")

data['sentiment_label'] = data['sentiment_label'].replace({'LABEL_1': '긍정', 'LABEL_0': '부정'})
sentiment_counts = data['sentiment_label'].value_counts()

fig, ax = plt.subplots(figsize=(6, 4))
ax.pie(
    sentiment_counts, 
    labels=sentiment_counts.index, 
    autopct='%1.1f%%', 
    colors=['skyblue', 'salmon'], 
    startangle=90
)
ax.set_title("AI 봇에 대한 사용자 반응", fontsize=16)

st.pyplot(fig)

# 3. AI 봇과 일반 사용자에 대한 답글 및 좋아요 수 비교
st.header("3. AI 봇과 일반 사용자에 대한 답글 및 좋아요 수 비교")

ai_bot_summary = ai_bot_data.agg({'Reply': 'count', 'Likes': 'sum'}).rename("AI 봇")
user_summary = user_data.agg({'Reply': 'count', 'Likes': 'sum'}).rename("일반 사용자")

comparison_df = pd.DataFrame([ai_bot_summary, user_summary]).T.reset_index()
comparison_df.columns = ['항목', 'AI 봇', '일반 사용자']

fig, ax = plt.subplots(figsize=(6, 4))
comparison_df.plot(
    x='항목',
    kind='bar',
    ax=ax,
    color=['skyblue', 'salmon'],
    edgecolor='black'
)

ax.set_title("AI 봇과 일반 사용자에 대한 답글 및 좋아요 수 비교", fontsize=16)
ax.set_xlabel("항목", fontsize=12)
ax.set_ylabel("수치", fontsize=12)
ax.legend(title="유형", fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

st.pyplot(fig)

# 4. 분야별 분석 (정치, 연예, 기술)
st.header("분야별 AI 봇과 일반 사용자 비교 분석")

categories = ['기술', '연예', '정치']

for category in categories:
    st.subheader(f"분야: {category}")

    category_data = data[data['Class'] == category]
    ai_bot_data = category_data[category_data['bot'] == 1]
    user_data = category_data[category_data['bot'] == 0]

    st.subheader(f"{category} 분야 - 감정 점수와 댓글 길이 분포")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ai_bot_data['comment_length'], ai_bot_data['sentiment_score'], color='skyblue', label='AI 봇', alpha=0.6)
    ax.scatter(user_data['comment_length'], user_data['sentiment_score'], color='salmon', label='일반 사용자', alpha=0.6)
    ax.set_title(f"{category} 분야 - 감정 점수와 댓글 길이 분포", fontsize=16)
    ax.set_xlabel("댓글 길이", fontsize=12)
    ax.set_ylabel("감정 점수", fontsize=12)
    ax.legend(title="유형", fontsize=10)
    st.pyplot(fig)

    st.subheader(f"{category} 분야 - 평균 감정 점수와 댓글 길이 비교")

    sentiment_means = {
        "AI 봇": ai_bot_data['sentiment_score'].mean(),
        "일반 사용자": user_data['sentiment_score'].mean()
    }
    length_means = {
        "AI 봇": ai_bot_data['comment_length'].mean(),
        "일반 사용자": user_data['comment_length'].mean()
    }

    sentiment_df = pd.DataFrame(sentiment_means.items(), columns=["유형", "평균 감정 점수"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=sentiment_df, x="유형", y="평균 감정 점수", palette=['skyblue', 'salmon'], ax=ax)
    ax.set_title(f"{category} 분야 - 평균 감정 점수", fontsize=16)
    ax.set_ylim(0, 1) 
    ax.set_xlabel("유형", fontsize=12)
    ax.set_ylabel("평균 감정 점수", fontsize=12)
    st.pyplot(fig)

    length_df = pd.DataFrame(length_means.items(), columns=["유형", "평균 댓글 길이"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=length_df, x="유형", y="평균 댓글 길이", palette=['skyblue', 'salmon'], ax=ax)
    ax.set_title(f"{category} 분야 - 평균 댓글 길이", fontsize=16)
    ax.set_xlabel("유형", fontsize=12)
    ax.set_ylabel("평균 댓글 길이", fontsize=12)
    st.pyplot(fig)
