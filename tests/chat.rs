//! The hand-compiled chat template against the real jinja one.
//!
//! Reference: tools/dump_chat.py
//!
//! Checked twice over: the rendered string character for character, and the
//! token ids it encodes to, so a whitespace slip that happens to tokenize the
//! same still fails.

use ornith_rs::chat::{Content, Message, Part, Role, apply_chat_template};
use ornith_rs::json::Value;
use ornith_rs::refs::{RefCase, model_dir};
use ornith_rs::tokenizer::Tokenizer;
use ornith_rs::tokenizer::split::Dialect;

#[test]
fn renders_like_jinja() {
    let Some(case) = RefCase::open_or_skip("chat") else { return };
    let cases = case.json("cases");
    let cases = cases.as_arr().unwrap();
    let rendered = case.strings("rendered");
    assert_eq!(cases.len(), rendered.len());

    // The reference ids come from mlx-lm, so they are transformers' dialect.
    let tk = model_dir()
        .exists()
        .then(|| Tokenizer::load_as(&model_dir(), Dialect::Transformers).unwrap());
    let ids = case.json("ids");
    let ids = ids.as_arr().unwrap();

    for (i, c) in cases.iter().enumerate() {
        let messages: Vec<Message> =
            c.get("messages").unwrap().as_arr().unwrap().iter().map(message).collect();
        let add_gen = c.get("add_generation_prompt").unwrap().as_bool().unwrap();
        let think = c.get("enable_thinking").unwrap().as_bool().unwrap();
        let got = apply_chat_template(&messages, add_gen, think)
            .unwrap_or_else(|e| panic!("case {i}: {e}"));
        assert_eq!(got, rendered[i], "case {i}");

        if let Some(tk) = &tk {
            let want: Vec<u32> =
                ids[i].as_arr().unwrap().iter().map(|v| v.as_usize().unwrap() as u32).collect();
            assert_eq!(tk.encode(&got), want, "case {i}: ids");
        }
    }
}

fn message(v: &Value) -> Message {
    let role = match v.get("role").and_then(|r| r.as_str()).unwrap() {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        r => panic!("unknown role {r}"),
    };
    let content = match v.get("content") {
        None | Some(Value::Null) => Content::None,
        Some(Value::Str(s)) => Content::Text(s.clone()),
        Some(Value::Arr(items)) => Content::Parts(items.iter().map(part).collect()),
        Some(other) => panic!("unexpected content {other:?}"),
    };
    Message {
        role,
        content,
        reasoning_content: v
            .get("reasoning_content")
            .and_then(|r| r.as_str())
            .map(|s| s.to_string()),
    }
}

/// The macro's own test order: image, then video, then text.
fn part(v: &Value) -> Part {
    let ty = v.get("type").and_then(|t| t.as_str());
    if v.get("image").is_some() || v.get("image_url").is_some() || ty == Some("image") {
        Part::Image
    } else if v.get("video").is_some() || ty == Some("video") {
        Part::Video
    } else if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
        Part::Text(t.to_string())
    } else {
        panic!("unexpected item type in content: {v:?}")
    }
}
