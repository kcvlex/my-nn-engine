use minijinja::context;
use minijinja::Environment;
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum ChatTemplateError {
    #[error("template error: {0}")]
    Template(#[from] minijinja::Error),
}

#[derive(Serialize, Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }
}

pub fn apply_chat_template(
    template_src: &str,
    messages: &[ChatMessage],
    eos_token: &str,
    bos_token: &str,
    add_generation_prompt: bool,
) -> Result<String, ChatTemplateError> {
    let mut env = Environment::new();
    env.add_template("chat", template_src)?;
    let tmpl = env.get_template("chat")?;
    let rendered = tmpl.render(context! {
        messages => messages,
        eos_token => eos_token,
        bos_token => bos_token,
        add_generation_prompt => add_generation_prompt,
    })?;
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINYLLAMA_TEMPLATE: &str = r#"{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|user|>
' + message['content'] + eos_token }}
{% elif message['role'] == 'system' %}
{{ '<|system|>
' + message['content'] + eos_token }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>
'  + message['content'] + eos_token }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{ '<|assistant|>' }}
{% endif %}
{% endfor %}"#;

    #[test]
    fn tinyllama_chat_template_renders() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hi."),
        ];
        let out = apply_chat_template(TINYLLAMA_TEMPLATE, &messages, "</s>", "<s>", true).unwrap();
        assert!(out.contains("<|system|>"));
        assert!(out.contains("You are helpful.</s>"));
        assert!(out.contains("<|user|>"));
        assert!(out.contains("Hi.</s>"));
        assert!(out.trim_end().ends_with("<|assistant|>"));
    }
}
