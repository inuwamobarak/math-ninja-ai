from django.shortcuts import render

from django.http import HttpResponse
# from .models import MathNinja

def math_ninja(request):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained("meta-math/MetaMath-Mistral-7B", torch_dtype=torch.bfloat16, device_map={"default": device})
    tokenizer = AutoTokenizer.from_pretrained("meta-math/MetaMath-Mistral-7B")

    def generate_response(prompt):
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            output = model.generate(inputs, max_length=400, pad_token_id=tokenizer.eos_token_id).to(device)
        response = tokenizer.decode(output[0])
        return response

    if request.method == 'POST':
        user_input = request.POST['user_input']
        response = generate_response(user_input)
        return HttpResponse(f"Response:\n{response}")
    else:
        return render(request, 'index.html', {'title': 'Math Ninja'})