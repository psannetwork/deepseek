from DeepSeekAPI import DeepSeekChat

# 一時的に直接ハードコードして確認
ds_session_id = "f9a84889488440b1a3e53bffe44e5828"
authorization_token = "Bearer dFjRQ1lzgfNrzdtvxbnC9qslM+mHIEfI09IwDKongXBekH+RqfytRWbC2uEZtJXv"

chat = DeepSeekChat(ds_session_id, authorization_token)
result = chat.send_message("Hello!", thinking_enabled=False)
print(result)

