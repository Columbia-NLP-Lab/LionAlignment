DPO_DATA_COLUMNS_TO_REMOVE = [
    "prompt",
    "prompt_id",
    "chosen",
    "rejected",
    "messages",
    "score_chosen",
    "score_rejected",
    "other_info",  # for storing other meta data information
]

DPO_DATA_MIX_COLUMNS = [
    'prompt',
    "prompt_id",
    'chosen',
    'rejected',
    'messages',
    'score_chosen',
    'score_rejected'
]


SFT_DATA_COLUMNS_TO_REMOVE = [
    "prompt",
    "prompt_id",
    "messages",
    "other_info",  # for storing other meta data information
]


OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0613-verbose",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13"
)


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


NO_SPECIAL_TOKEN_CHAT_TEMPLATE = "{% if messages[0]['content'] == '' %}" \
"{% set loop_messages = messages[1:] %}" \
"{% else %}" \
"{% set loop_messages = messages %}" \
"{% endif %}" \
"{% for message in loop_messages %}" \
"{% if message['role'] == 'user' %}" \
"{{ 'User: ' + message['content'].strip() + ' '}}" \
"{% elif message['role'] == 'system' %}" \
"{{ 'System: ' + message['content'].strip() + ' '}}" \
"{% elif message['role'] == 'assistant' %}" \
"{{ 'Assistant: ' + message['content'] + eos_token}}" \
"{% endif %}" \
"{% endfor %}"


ULTRALM_CHAT_TEMPLATE = "{% if messages[0]['content'] == '' %}" \
"{% set loop_messages = messages[1:] %}" \
"{% else %}" \
"{% set loop_messages = messages %}" \
"{% endif %}" \
"{% for message in loop_messages %}" \
"{% if message['role'] == 'user' %}" \
"{{ 'User: ' + message['content'].strip() + '\n'}}" \
"{% elif message['role'] == 'system' %}" \
"{{ 'System: ' + message['content'].strip() + '\n'}}" \
"{% elif message['role'] == 'assistant' %}" \
"{{ 'Assistant: ' + message['content'] + eos_token + '\n'}}" \
"{% endif %}" \
"{% endfor %}"


VICUNA_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}" \
"{% set loop_messages = messages[1:] %}" \
"{% set system_message = messages[0]['content'].strip() + ' ' %}" \
"{% else %}" \
"{% set loop_messages = messages %}" \
"{% set system_message = '' %}" \
"{% set system_message = '' %}" \
"{% endif %}" \
"{{ bos_token + system_message }}" \
"{% for message in loop_messages %}" \
"{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}" \
"{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}" \
"{% endif %}" \
"{% if message['role'] == 'user' %}" \
"{{ 'USER: ' + message['content'].strip() }}" \
"{% elif message['role'] == 'assistant' %}" \
"{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}" \
"{% endif %}" \
"{% if loop.last and message['role'] == 'user' and add_generation_prompt %}" \
"{{ ' ASSISTANT:' }}" \
"{% endif %}" \
"{% endfor %}"


ALPACA_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}" \
"{% set loop_messages = messages[1:] %}" \
"{% set system_message = messages[0]['content'].strip() + '\n' %}" \
"{% else %}" \
"{% set loop_messages = messages %}" \
"{% set system_message = '' %}" \
"{% endif %}" \
"{{ bos_token + system_message }}" \
"{% for message in loop_messages %}" \
"{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}" \
"{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}" \
"{% endif %}" \
"{% if message['role'] == 'user' %}" \
"{{ '### Instruction:\n' + message['content'].strip() + '\n' }}" \
"{% elif message['role'] == 'assistant' %}" \
"{{ '### Response:\n' + message['content'].strip() + eos_token + '\n' }}" \
"{% endif %}" \
"{% if loop.last and message['role'] == 'user' and add_generation_prompt %}" \
"{{ '### Instruction:\n' }}" \
"{% endif %}" \
"{% endfor %}"


STAR_CHAT_TEMPLATE = "{% if messages[0]['content'] == '' %}" \
"{% set loop_messages = messages[1:] %}" \
"{% else %}" \
"{% set loop_messages = messages %}" \
"{% endif %}" \
"{% for message in loop_messages %}" \
"{% if message['role'] == 'user' %}" \
"{{ '<|user|>\n' + message['content'].strip() + '<|end|>'}}" \
"{% elif message['role'] == 'system' %}" \
"{{ '<|system|>\n' + message['content'].strip() + '<|end|>'}}" \
"{% elif message['role'] == 'assistant' %}" \
"{{ '<|assistant|>\n' + message['content'] + '<|end|>'}}" \
"{% endif %}" \
"{% endfor %}"