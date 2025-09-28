import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Agente de Análise de CSV",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agente Autônomo para Análise de Dados em CSV")
st.write("""
Esta aplicação utiliza um agente de IA para responder perguntas sobre arquivos CSV. 
Para começar, faça o upload do seu arquivo CSV na barra lateral e comece a conversar!
""")

# --- Funções Auxiliares ---
def carregar_e_processar_csv(arquivo_csv):
    """Carrega um arquivo CSV em um DataFrame do Pandas."""
    try:
        df = pd.read_csv(arquivo_csv)
        return df
    except UnicodeDecodeError:
        st.warning("Falha na decodificação UTF-8. Tentando com 'latin1'.")
        arquivo_csv.seek(0)
        df = pd.read_csv(arquivo_csv, encoding='latin1')
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo CSV: {e}")
        return None

# --- Inicialização do Estado da Sessão ---
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Lógica da API Key (com suporte para Streamlit Secrets) ---
try:
    st.session_state.google_api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    st.sidebar.warning("A chave da API do Google não foi encontrada nos segredos. Por favor, insira-a abaixo.")
    api_key_input = st.sidebar.text_input(
        "Chave da API do Google", 
        type="password", 
        help="Insira sua chave aqui para rodar localmente."
    )
    if api_key_input:
        st.session_state.google_api_key = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key_input
        st.sidebar.success("API Key configurada!")

# --- Barra Lateral (Sidebar) para Upload ---
with st.sidebar:
    st.header("Upload do Arquivo")
    arquivo_csv = st.file_uploader("Selecione um arquivo CSV", type=["csv"])

    if arquivo_csv:
        st.session_state.df = carregar_e_processar_csv(arquivo_csv)
        if st.session_state.df is not None:
            st.success("Arquivo CSV carregado!")
            st.dataframe(st.session_state.df.head(), use_container_width=True)

# --- Lógica Principal da Aplicação ---
if st.session_state.google_api_key and st.session_state.df is not None:
    
    if st.session_state.agent is None:
        st.info("Inicializando o agente de IA... Isso pode levar um momento.")
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro-latest",
                temperature=0,
                convert_system_message_to_human=True
            )
            
            # --- CORREÇÃO PRINCIPAL AQUI ---
            st.session_state.agent = create_pandas_dataframe_agent(
                llm=llm,
                df=st.session_state.df,
                # Usar a string 'tool-calling', que o erro confirmou ser suportada
                agent_type='tool-calling', 
                verbose=True,
                handle_parsing_errors=True,
                agent_executor_kwargs={"handle_parsing_errors": True},
                allow_dangerous_code=True
            )
            st.success("Agente pronto para conversar!")
        except Exception as e:
            st.error(f"Erro ao criar o agente: {e}")
            st.stop()

    st.header("Converse com seus Dados")

    # Adiciona a mensagem inicial do assistente se o chat estiver vazio
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Olá! Sou seu assistente de análise de dados. O que você gostaria de saber sobre este arquivo?",
            "figure": None
        })

    # Exibe o histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "figure" in message and message["figure"] is not None:
                st.pyplot(message["figure"])

    # Captura a pergunta do usuário
    if prompt := st.chat_input("Qual a distribuição da variável 'idade'?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("O agente está pensando..."):
                try:
                    plt.close('all') # Garante que figuras antigas sejam limpas
                    
                    # O agente agora precisa do histórico de chat para ser conversacional
                    chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

                    response = st.session_state.agent.invoke({
                        "input": prompt,
                        "chat_history": chat_history # Passa o histórico para o agente
                    })
                    
                    output_text = response["output"]
                    
                    fig = plt.gcf()
                    has_plot = any(ax.has_data() for ax in fig.get_axes()) if fig else False

                    if has_plot:
                        st.pyplot(fig)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": output_text, 
                            "figure": fig
                        })
                    else:
                        plt.close(fig)
                        st.markdown(output_text)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": output_text, 
                            "figure": None
                        })

                except Exception as e:
                    error_message = f"Ocorreu um erro: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message, "figure": None})

else:
    st.info("Por favor, configure a API Key e faça o upload de um arquivo CSV na barra lateral para começar.")
