# Comparação de Modelos para Text Ranking

- Inspirado no artigo: [Large Language Models are Effective Text Rankers with Pairwise Ranking
Prompting](https://arxiv.org/pdf/2306.17563)

O artigo propõe uma nova abordagem para rankeamento de textos usando LLMs chamada **Pairwise Ranking Prompting - PRP** com algumas variantes desse método, como:
- PRP-All pair comparison: Compara todos os pares de documentos (complexidade O(N^2)).
- PRP-Sorting: Usa algoritmos de ordenação como Heapsort (O(N log N)).
- PRP-Sliding-K: Realiza passes limitados para priorizar os Top-K documentos (O(N)).

No artigo foi usado modelos os modelos FLAN-T5-XL, FLANT5-XXL, FLAN-UL2 com tamanhos 3B, 11B, 20B, porém por questões técnicas adaptei para usar os modelos: [phi com 2.7B de parâmetros](https://ollama.com/library/phi) e [deepseek R1 com 1.5B de parâmetros](https://ollama.com/library/deepseek-r1). 

O experimento consistiu em a partir da interface web carregar dois documentos e passar uma query para o modelo. No caso, a mesma query, documentos e prompt foram usadas para os dois modelos.

Na comparação entre dos documentos os dois modelos chegaram na resposta correta, porém o modelo phi de forma mais direta e concisa na resposta. Conforme observado nas imagens:

- phi
  
![Resposta do modelo Phi](https://github.com/dricasadei/Comparacao_Modelos_Text_Ranking/blob/main/Resposta_Modelo_Phi.png)

- deepseek R1
  
![Resposta do modelo DeepSeek R1 - Parte 1](https://github.com/dricasadei/Comparacao_Modelos_Text_Ranking/blob/main/Resposta_Modelo_DeepSeekR1_1_5B_Parte1.png)
![Resposta do modelo DeepSeek R1 - Parte 2](https://github.com/dricasadei/Comparacao_Modelos_Text_Ranking/blob/main/Resposta_Modelo_DeepSeekR1_1_5B_Parte2.png)

Ferramentas utilizadas:
- [Plataforma Ollama](https://ollama.com/)
- Python
- [Langchain](https://api.python.langchain.com/en/latest/langchain_api_reference.html)
- [Chroma](https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma)
- [Streamlit](https://streamlit.io/)
