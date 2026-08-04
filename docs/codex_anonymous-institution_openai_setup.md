# Codex anonymous-institution OpenAI Setup

Use this when you want Codex to run through the Uni anonymous-institution AI Gateway with the OpenAI models that currently work best with Codex.

## 1. Add the Codex config

Run this once:

```bash
mkdir -p ~/.codex && cat >> ~/.codex/config.toml <<'EOF'

[model_providers.anonymous-institution_gateway]
name = "Uni anonymous-institution AI Gateway"
base_url = "https://ai-gateway.anonymous-institution.de/v1"
wire_api = "responses"
env_key = "anonymous-institution_AI_GATEWAY_API_KEY"

[profiles.anonymous-institution-gpt-5-4]
model_provider = "anonymous-institution_gateway"
model = "openai.gpt-5.4"

[profiles.anonymous-institution-gpt-5-4-mini]
model_provider = "anonymous-institution_gateway"
model = "openai.gpt-5.4-mini"

[profiles.anonymous-institution-gpt-5-3-codex]
model_provider = "anonymous-institution_gateway"
model = "openai.gpt-5.3-codex"

[profiles.anonymous-institution-gpt-5-2-codex]
model_provider = "anonymous-institution_gateway"
model = "openai.gpt-5.2-codex"

[profiles.anonymous-institution-gpt-5-1-codex]
model_provider = "anonymous-institution_gateway"
model = "openai.gpt-5.1-codex"

[profiles.anonymous-institution-gpt-5-2]
model_provider = "anonymous-institution_gateway"
model = "openai.gpt-5.2"

[profiles.anonymous-institution-gpt-5]
model_provider = "anonymous-institution_gateway"
model = "openai.gpt-5"
EOF
```

## 2. Export your gateway key

```bash
export anonymous-institution_AI_GATEWAY_API_KEY="PASTE_KEY_HERE"
```

To make it persistent, add the same line to your `~/.bashrc` or `~/.zshrc`.

## 3. Start Codex with a profile

Examples:

```bash
codex --profile anonymous-institution-gpt-5-4
```

```bash
codex --profile anonymous-institution-gpt-5-3-codex
```

```bash
codex --profile anonymous-institution-gpt-5-2-codex
```

## 4. Quick test

```bash
codex exec --profile anonymous-institution-gpt-5-4 "How old is the universe?"
```

## Notes

- Stick to the `openai.*` profiles for now.
- The `gwdg.*` models currently fail for Codex because the gateway/backend does not fully support the message/tool format Codex sends.
- `codex resume --last --profile anonymous-institution-gpt-5-4` also works if you want to resume using a specific profile.
