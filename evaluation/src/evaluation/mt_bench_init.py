from fastchat.conversation import (
    register_conv_template,
    Conversation, SeparatorStyle
)
from fastchat.model.model_adapter import (
    register_model_adapter, get_conv_template,
    BaseModelAdapter,
    model_adapters
)


### add in new model templates
### Gemma
class GemmaAdapter(BaseModelAdapter):
    """The model adapter for Gemma"""

    def match(self, model_path: str):
        return "templ_gemma" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("templ_gemma")


### ChatML generic
class ChatMLAdapter(BaseModelAdapter):
    """The model adapter for models that chat with chatML"""

    def match(self, model_path: str):
        return "templ_chatml" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("templ_chatml")


### H4 Zephyr Beta
class ZephyrMistralAdapter(BaseModelAdapter):
    """The model adapter for Zephyr (e.g. HuggingFaceH4/zephyr-7b-beta)"""

    def match(self, model_path: str):
        return "templ_zephyr" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("templ_zephyr")


def register_model_adapter(cls):
    model_adapters.insert(0, cls())  # prioritize new adapters


def register_all():
    """Register all model adapters"""
    # gemma
    register_conv_template(
        Conversation(
            name="templ_gemma",
            system_message="<bos>",
            roles=("<start_of_turn>user\n", "<start_of_turn>model\n"),
            sep_style=SeparatorStyle.NO_COLON_SINGLE,
            sep="<end_of_turn>\n",
            stop_str="<end_of_turn>",
        ),
        override=True
    )
    register_model_adapter(GemmaAdapter)

    # chatML
    register_conv_template(
        Conversation(
            name="templ_chatml",
            system_template="<bos><|im_start|>system\n{system_message}",
            system_message="You are an AI assistant.",
            roles=("<|im_start|>user", "<|im_start|>assistant"),
            sep_style=SeparatorStyle.CHATML,
            sep="<|im_end|>",
            stop_str=["<|im_end|>", "<|endoftext|>"],
        ),
        override=True
    )
    register_model_adapter(ChatMLAdapter)

    # zephyr
    register_conv_template(
        Conversation(
            name="templ_zephyr",
            system_template="<|system|>\n",
            roles=("<|user|>", "<|assistant|>"),
            sep_style=SeparatorStyle.CHATML,
            sep="</s>",
            stop_token_ids=[2],
            stop_str="</s>",
        ),
        override=True
    )
    register_model_adapter(ZephyrMistralAdapter)

    # return available templates
    return [
        'templ_gemma',
        'templ_chatml',
        'templ_zephyr'
    ]