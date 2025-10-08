import streamlit as st
import ollama

model = "gemma3"
st.title(model)

if "message" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def generate_response():
    response = ollama.chat(model=model, stream = True, messages = st.session_state.messages)# since we deifined stream=True, the response is a generator or response is in chunks
    for chunk in response:
        token = chunk["message"]["content"]
        st.session_state["full_messgae"] += token
        yield token

if prompt := st.chat_input("Ready for your request: "):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state["full_messgae"] = ""
    with st.chat_message("assistant"):
        stream = generate_response()
        response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
    