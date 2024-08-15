# CityU STEM Challenge 2024 - iAsk
## 安装
### 第三方服務 API Key 要求
1. OpenAI 或其他 LLMs API key
2. 網頁密碼
3. Serpapi API key 
4. 以及代理地址（如有）
   
   
``` 
# Python 3
git clone https://github.com/vc08932/streamlit-LLM # Download the code
cd streamlit-LLM 
pip install -r requirements.txt 
python.exe setup.py # Configure the required key
```

## 使用
1. 運行：`streamlit run index.py` 
2. 檢查瀏覽器，如無網頁自動開啟，輸入 `http://localhost:8501` 進入網頁
3. 如欲結束運行，在终端按下 `Ctrl + C` 結束程序

## 程序架構
```
.
├── setup.py
└── index.py
    ├── pages
    │   ├── chatbot.py
    │   └── quiz.py
    ├── src
    │   ├── llm_caller.py
    │   ├── search.py
    │   └── quiz_question.json
    └── .streamlit
        ├── config.toml
        └── secrets.toml
```
