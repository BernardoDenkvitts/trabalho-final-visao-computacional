# 📘 Analisador de Imagens — Front-end + API FastAPI

Este projeto é composto por:

* **Uma interface web estática (HTML/CSS/JS)** para upload e visualização de imagens processadas.
* **Uma API em FastAPI** responsável por:

  * Receber uma imagem (`UploadFile`)
  * Processá-la localmente (ex.: segmentação)
  * Enviar a imagem para um serviço remoto (modelo externo)
  * Retornar para o front:

    * `label`
    * `mensagem` / `message`
    * `imagem` processada (data URL)
    * `raw_remote_response` para debug

---

## 🚀 Funcionalidades

### **Front-end**

* Upload por clique ou drag & drop
* Preview da imagem enviada
* Endpoint configurável direto na interface
* Exibição:

  * Categoria (`label`)
  * Mensagem retornada
  * Imagem processada (base64)
  * Resposta completa (modo debug)
* Feedback visual com loader e erros estilizados

### **Back-end (FastAPI)**

* Endpoint: `POST /analyze`
* Recebe arquivos multipart (`image`)
* Converte, processa e envia a imagem para outro serviço externo (via httpx)
* Retenta requisição em caso de falha
* Proteção contra erros comuns (JSON inválido, timeout, FileNotFound)
* Totalmente compatível com CORS para testes locais


# 🧩 Como rodar localmente

## 1️⃣ Rodar a API (FastAPI)

### Instale dependências:

```bash
pip install -r requirements.txt
```

### Execute o servidor:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

A API ficará disponível em:

```
http://127.0.0.1:8000
```

## 2️⃣ Rodar o Front-end

### Opção A — Servidor Python simples:

```bash
cd app
python -m http.server 8080
```

Abra o navegador em:

```
http://0.0.0.0:8080
```

# 🧪 Fluxo de Funcionamento

1. Usuário envia uma imagem pelo front-end
2. O navegador envia via `POST multipart/form-data` para `/analyze`
3. O FastAPI:

   * Lê o arquivo
   * Executa `segment_from_upload_bytes`
   * Envia para `ANALYSIS_API_URL` (modelo externo)
   * Recebe JSON de resposta
   * Gera uma imagem processada em Base64
   * Retorna para o front-end:

     ```json
     {
       "label": "...",
       "mensagem": "...",
       "imagem": "data:image/png;base64,...",
       "raw_remote_response": {...}
     }
     ```
4. O front exibe tudo na interface.

---

# 📸 Exemplo de Resposta

```json
{
  "label": "defeito_detectado",
  "mensagem": "Área com anomalias identificada.",
  "imagem": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
}
