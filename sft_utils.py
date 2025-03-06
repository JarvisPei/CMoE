from peft import get_peft_model, LoraConfig, TaskType

from datautils import *

def simple_sft(model, args, epoch = 1):
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, bsz = args.sft_bsz
    )


    lora_config = LoraConfig(
        r = 8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    lr = 5.65e-5
    wd = 0
    betas = (0.9, 0.95)
    eps = 1e-8

    extra_params = []
    base_params = []

    for name, param in model.named_parameters():
        if 'extra' in name:
            extra_params.append(param)
        else:
            base_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': lr, 'betas': betas, 'eps': eps, 'wd': wd},
        {'params': extra_params, 'lr': args.extra_lr, 'betas': betas, 'eps': eps, 'wd': wd}
    ])

    num_epoch=epoch
    model.train()
    for epoch in range(num_epoch):
        epoch_loss = 0
        for batch in dataloader:
            outputs = model(batch[0].to('cuda'), labels = batch[0].to('cuda'))
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
    
        avg_loss = epoch_loss/len(dataloader)
        print(f'avg_loss:{avg_loss}')

    model = model.merge_and_unload()

    return model