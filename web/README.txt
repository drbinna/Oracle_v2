Quant Firm — ready-to-deploy bundle (Vercel)

Files:
  index.html                -> the landing (served at / )
  quant_firm_viewer.html    -> the verification console + judge chatbot
  api/ask.py                -> serverless chat endpoint (/api/ask), stdlib only

Deploy (3 commands):
  npm i -g vercel
  vercel                       # from inside this folder; accept the prompts
  vercel env add LLM_API_KEY   # paste your Fireworks key, choose Production
  vercel --prod                # redeploy to production with the key live

Optional env vars (set the same way):
  LLM_MODEL      e.g. accounts/fireworks/models/llama-v3p3-70b-instruct
  LLM_BASE_URL   default https://api.fireworks.ai/inference/v1
  -- to run on the model you trained instead:
       LLM_BASE_URL = https://inference.beta.hud.ai/v1
       LLM_API_KEY  = <your HUD key>
       LLM_MODEL    = Qwen/Qwen3-8B

The viewer calls /api/ask on the same origin, so there is no CORS to configure.
Your key lives only in Vercel's encrypted env — never in the files.
