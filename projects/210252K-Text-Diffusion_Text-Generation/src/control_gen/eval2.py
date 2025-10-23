def eval_ppl(args, text_samples):
    '''
    Evaluating using GPT2 finetuned on this task
    '''
    print(f'loading model from {args.model_name_or_path}')
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
        ).cuda()
        print(f"Successfully loaded model from {args.model_name_or_path}")
    except Exception as e:
        print(f"Could not load model from {args.model_name_or_path}")
        print(f"Error: {e}")
        print(f"Using gpt2 instead")
        model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()
        args.model_name_or_path = "gpt2"

    # Load tokenizer
    print(f'loading tokenizer from {args.model_name_or_path}')
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        print(f"Successfully loaded tokenizer from {args.model_name_or_path}")
    except Exception as e:
        print(f"Could not load tokenizer from {args.model_name_or_path}")
        print(f"Error: {e}")
        print(f"Using gpt2 tokenizer instead")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print('finished loading models.')

    full_score = []
    skipped = 0
    
    # Handle both list (from 'file' format) and dict (from 'paired' format)
    if isinstance(text_samples, dict):
        samples_to_process = text_samples.items()
    else:
        samples_to_process = [(i, [sample]) for i, sample in enumerate(text_samples)]

    for gold, full_word_lst in samples_to_process:
        agg_loss = []
        for x in full_word_lst:
            # Join tokens into text - match training format exactly
            text = " ".join(x)
            
            # Add BOS/EOS with spaces - MUST match training format
            text = tokenizer.bos_token + " " + text + " " + tokenizer.eos_token
            
            # Tokenize
            tokenized_x = tokenizer(
                text, 
                return_tensors='pt',
                truncation=True,
                max_length=512  # Prevent OOM on very long sequences
            )
            
            input_ids = tokenized_x['input_ids'].cuda()
            
            # Skip if sequence is too short (just BOS/EOS)
            if input_ids.shape[1] <= 2:
                skipped += 1
                continue
            
            labels = input_ids.clone()
            
            # Calculate loss
            with torch.no_grad():  # Save memory
                model_output = model(input_ids, labels=labels)
            
            agg_loss.append(model_output.loss.item())
        
        if agg_loss:  # Only add if we have valid losses
            example_mean_score = torch.tensor(agg_loss).mean()
            full_score.append(example_mean_score)
    
    if not full_score:
        print("ERROR: No valid samples to evaluate!")
        return
    
    full_score_ = np.array(full_score).mean()
    full_score_std = np.array(full_score).std()
    
    print(f'\n{"="*50}')
    print(f'Perplexity Evaluation Results:')
    print(f'{"="*50}')
    print(f'Number of samples evaluated: {len(full_score)}')
    print(f'Number of samples skipped: {skipped}')
    print(f'Average NLL: {full_score_:.4f} ± {full_score_std:.4f}')
    print(f'Average PPL: {np.exp(full_score_):.4f}')
    print(f'{"="*50}\n')
    
    return full_score_