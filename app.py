{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import pandas as pd\
import os\
from langchain_google_genai import ChatGoogleGenerativeAI\
from langchain_experimental.agents import create_pandas_dataframe_agent\
from langchain.agents.agent_types import AgentType\
from langchain.memory import ConversationBufferMemory\
\
# --- Configura\'e7\'e3o da P\'e1gina Streamlit ---\
st.set_page_config(\
    page_title="Agente de An\'e1lise de CSV",\
    page_icon="\uc0\u55358 \u56598 ",\
    layout="wide"\
)\
\
st.title("\uc0\u55358 \u56598  Agente Aut\'f4nomo para An\'e1lise de Dados em CSV")\
st.write("""\
Esta aplica\'e7\'e3o utiliza um agente de IA para responder perguntas sobre arquivos CSV. \
Para come\'e7ar:\
1. Insira sua chave da API do Google.\
2. Fa\'e7a o upload do seu arquivo CSV.\
3. Comece a conversar com o agente!\
""")\
\
# --- Fun\'e7\'f5es Auxiliares ---\
\
def carregar_e_processar_csv(arquivo_csv):\
    """Carrega um arquivo CSV em um DataFrame do Pandas."""\
    try:\
        # Tenta ler com codifica\'e7\'e3o padr\'e3o\
        df = pd.read_csv(arquivo_csv)\
        return df\
    except UnicodeDecodeError:\
        # Se falhar, tenta com 'latin1', comum em arquivos brasileiros\
        st.warning("Falha na decodifica\'e7\'e3o UTF-8. Tentando com 'latin1'.")\
        arquivo_csv.seek(0) # Retorna ao in\'edcio do arquivo\
        df = pd.read_csv(arquivo_csv, encoding='latin1')\
        return df\
    except Exception as e:\
        st.error(f"Erro ao carregar o arquivo CSV: \{e\}")\
        return None\
\
# --- Inicializa\'e7\'e3o do Estado da Sess\'e3o ---\
# Usamos o session_state do Streamlit para manter os dados entre as intera\'e7\'f5es\
\
if 'google_api_key' not in st.session_state:\
    st.session_state.google_api_key = None\
if 'df' not in st.session_state:\
    st.session_state.df = None\
if 'agent' not in st.session_state:\
    st.session_state.agent = None\
if "messages" not in st.session_state:\
    st.session_state.messages = []\
\
# --- Barra Lateral (Sidebar) para Configura\'e7\'f5es ---\
with st.sidebar:\
    st.header("1. Configura\'e7\'e3o da API")\
    # Campo para a chave da API\
    api_key_input = st.text_input(\
        "Chave da API do Google", \
        type="password", \
        help="Voc\'ea pode obter sua chave no Google AI Studio."\
    )\
\
    if api_key_input:\
        st.session_state.google_api_key = api_key_input\
        os.environ["GOOGLE_API_KEY"] = api_key_input\
        st.success("API Key configurada!")\
\
    st.header("2. Upload do Arquivo")\
    # Campo para upload do CSV\
    arquivo_csv = st.file_uploader("Selecione um arquivo CSV", type=["csv"])\
\
    if arquivo_csv:\
        st.session_state.df = carregar_e_processar_csv(arquivo_csv)\
        if st.session_state.df is not in None:\
            st.success("Arquivo CSV carregado com sucesso!")\
            st.dataframe(st.session_state.df.head(), use_container_width=True)\
\
# --- L\'f3gica Principal da Aplica\'e7\'e3o ---\
\
# Verifica se os pr\'e9-requisitos (API e CSV) foram atendidos\
if st.session_state.google_api_key and st.session_state.df is not None:\
    \
    # Cria o agente apenas uma vez\
    if st.session_state.agent is None:\
        try:\
            # Inicializa o modelo de linguagem (LLM)\
            llm = ChatGoogleGenerativeAI(\
                model="gemini-pro",\
                temperature=0, # Temperatura 0 para respostas mais determin\'edsticas\
                convert_system_message_to_human=True\
            )\
\
            # Inicializa a mem\'f3ria da conversa\
            memory = ConversationBufferMemory(\
                memory_key='chat_history', \
                return_messages=True,\
                input_key='input',\
                output_key='output'\
            )\
\
            # Cria o agente Pandas DataFrame\
            st.session_state.agent = create_pandas_dataframe_agent(\
                llm=llm,\
                df=st.session_state.df,\
                agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,\
                verbose=True, # Mostra os "pensamentos" do agente no terminal\
                memory=memory,\
                handle_parsing_errors=True, # Ajuda a lidar com erros de formata\'e7\'e3o do LLM\
                agent_executor_kwargs=\{"handle_parsing_errors": True\},\
                allow_dangerous_code=True # Necess\'e1rio para o agente executar c\'f3digo Python\
            )\
            st.success("Agente pronto para conversar!")\
        except Exception as e:\
            st.error(f"Erro ao criar o agente: \{e\}")\
            st.stop()\
\
    # --- Interface de Chat ---\
    st.header("3. Converse com seus Dados")\
\
    # Exibe o hist\'f3rico de mensagens\
    for message in st.session_state.messages:\
        with st.chat_message(message["role"]):\
            st.markdown(message["content"])\
            # Se a mensagem tiver um gr\'e1fico, exibe-o\
            if "figure" in message and message["figure"] is not None:\
                st.pyplot(message["figure"])\
\
    # Captura a pergunta do usu\'e1rio\
    if prompt := st.chat_input("Fa\'e7a uma pergunta sobre o seu arquivo CSV..."):\
        # Adiciona a pergunta do usu\'e1rio ao hist\'f3rico\
        st.session_state.messages.append(\{"role": "user", "content": prompt\})\
        with st.chat_message("user"):\
            st.markdown(prompt)\
\
        # Gera a resposta do agente\
        with st.chat_message("assistant"):\
            with st.spinner("O agente est\'e1 pensando..."):\
                try:\
                    # Limpa a figura anterior para evitar que seja exibida novamente\
                    st.pyplot(None)\
                    \
                    # Define o matplotlib como backend para gr\'e1ficos\
                    # Isso garante que o agente possa criar e salvar gr\'e1ficos\
                    plt_code = "import matplotlib.pyplot as plt; plt.figure()"\
                    \
                    # Executa o agente com a pergunta do usu\'e1rio\
                    response = st.session_state.agent.invoke(\{"input": f"\{prompt\}\\n\\n\{plt_code\}"\})\
                    \
                    # Extrai a resposta e a figura (se houver)\
                    output_text = response["output"]\
                    \
                    # O agente do LangChain pode gerar gr\'e1ficos. Capturamos a figura atual.\
                    fig = plt.gcf()\
                    # Verifica se a figura n\'e3o est\'e1 vazia\
                    has_plot = any(ax.has_data() for ax in fig.get_axes()) if fig else False\
\
                    if has_plot:\
                        st.pyplot(fig)\
                        st.session_state.messages.append(\{\
                            "role": "assistant", \
                            "content": output_text, \
                            "figure": fig\
                        \})\
                    else:\
                        plt.close(fig) # Fecha a figura vazia\
                        st.markdown(output_text)\
                        st.session_state.messages.append(\{\
                            "role": "assistant", \
                            "content": output_text, \
                            "figure": None\
                        \})\
\
                except Exception as e:\
                    error_message = f"Ocorreu um erro: \{e\}"\
                    st.error(error_message)\
                    st.session_state.messages.append(\{"role": "assistant", "content": error_message, "figure": None\})\
\
else:\
    # Mensagem de aviso se os pr\'e9-requisitos n\'e3o forem atendidos\
    st.warning("Por favor, configure a API Key e fa\'e7a o upload de um arquivo CSV na barra lateral para come\'e7ar.")\
\
}