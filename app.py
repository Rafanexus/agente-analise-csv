import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Agente de Análise de CSV",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agente Autônomo para Análise de Dados em CSV")
st.write("""
Esta aplicação utiliza um agente de IA para responder perguntas sobre arquivos CSV. 
Para começar:
1. Insira sua chave da API do Google.
2. Faça o upload do seu arquivo CSV.
3. Comece a conversar com o agente!
""")

# --- Funções Auxiliares ---

def carregar_e_processar_csv(arquivo_csv):
    """Carrega um arquivo CSV em um DataFrame do Pandas."""
    try:
        # Tenta ler com codificação padrão
        df = pd.read_csv(arquivo_csv)
        return df
    except UnicodeDecodeError:
        # Se falhar, tenta com 'latin1', comum em arquivos brasileiros
        st.warning("Falha na decodificação UTF-8. Tentando com 'latin1'.")
        arquivo_csv.seek(0) # Retorna ao início do arquivo
        df = pd.read_csv(arquivo_csv, encoding='latin1')
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo CSV: {e}")
        return None

# --- Inicialização do Estado da Sessão ---
# Usamos o session_state do Streamlit para manter os dados entre as interações

if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Barra Lateral (Sidebar) para Configurações ---
with st.sidebar:
    st.header("1. Configuração da API")
    # Campo para a chave da API
    api_key_input = st.text_input(
        "Chave da API do Google", 
        type="password", 
        help="Você pode obter sua chave no Google AI Studio."
    )

    if api_key_input:
        st.session_state.google_api_key = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key_input
        st.success("API Key configurada!")

    st.header("2. Upload do Arquivo")
    # Campo para upload do CSV
    arquivo_csv = st.file_uploader("Selecione um arquivo CSV", type=["csv"])

    if arquivo_csv:
        st.session_state.df = carregar_e_processar_csv(arquivo_csv)
        if st.session_state.df is not in None:
            st.success("Arquivo CSV carregado com sucesso!")
            st.dataframe(st.session_state.df.head(), use_container_width=True)

# --- Lógica Principal da Aplicação ---

# Verifica se os pré-requisitos (API e CSV) foram atendidos
if st.session_state.google_api_key and st.session_state.df is not None:
    
    # Cria o agente apenas uma vez
    if st.session_state.agent is None:
        try:
            # Inicializa o modelo de linguagem (LLM)
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0, # Temperatura 0 para respostas mais determinísticas
                convert_system_message_to_human=True
            )

            # Inicializa a memória da conversa
            memory = ConversationBufferMemory(
                memory_key='chat_history', 
                return_messages=True,
                input_key='input',
                output_key='output'
            )

            # Cria o agente Pandas DataFrame
            st.session_state.agent = create_pandas_dataframe_agent(
                llm=llm,
                df=st.session_state.df,
                agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                verbose=True, # Mostra os "pensamentos" do agente no terminal
                memory=memory,
                handle_parsing_errors=True, # Ajuda a lidar com erros de formatação do LLM
                agent_executor_kwargs={"handle_parsing_errors": True},
                allow_dangerous_code=True # Necessário para o agente executar código Python
            )
            st.success("Agente pronto para conversar!")
        except Exception as e:
            st.error(f"Erro ao criar o agente: {e}")
            st.stop()

    # --- Interface de Chat ---
    st.header("3. Converse com seus Dados")

    # Exibe o histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Se a mensagem tiver um gráfico, exibe-o
            if "figure" in message and message["figure"] is not None:
                st.pyplot(message["figure"])

    # Captura a pergunta do usuário
    if prompt := st.chat_input("Faça uma pergunta sobre o seu arquivo CSV..."):
        # Adiciona a pergunta do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gera a resposta do agente
        with st.chat_message("assistant"):
            with st.spinner("O agente está pensando..."):
                try:
                    # Limpa a figura anterior para evitar que seja exibida novamente
                    st.pyplot(None)
                    
                    # Define o matplotlib como backend para gráficos
                    # Isso garante que o agente possa criar e salvar gráficos
                    plt_code = "import matplotlib.pyplot as plt; plt.figure()"
                    
                    # Executa o agente com a pergunta do usuário
                    response = st.session_state.agent.invoke({"input": f"{prompt}\n\n{plt_code}"})
                    
                    # Extrai a resposta e a figura (se houver)
                    output_text = response["output"]
                    
                    # O agente do LangChain pode gerar gráficos. Capturamos a figura atual.
                    fig = plt.gcf()
                    # Verifica se a figura não está vazia
                    has_plot = any(ax.has_data() for ax in fig.get_axes()) if fig else False

                    if has_plot:
                        st.pyplot(fig)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": output_text, 
                            "figure": fig
                        })
                    else:
                        plt.close(fig) # Fecha a figura vazia
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
    # Mensagem de aviso se os pré-requisitos não forem atendidos
    st.warning("Por favor, configure a API Key e faça o upload de um arquivo CSV na barra lateral para começar.")

