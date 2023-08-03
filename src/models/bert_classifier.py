import torch
import collections


class CustomBertClassifier(torch.nn.Module):
    def __init__(self, base_model, hidden_layer, base_model_output_size=768, dropout=0.5, classifier=None):
        super().__init__()
        self.base_model = base_model

        if classifier is None:
            self.classifier = CustomClassifier(hidden_layer=hidden_layer,
                                               input_size=base_model_output_size,
                                               dropout=dropout).classifier
        else:
            self.classifier = classifier

    def forward(self, inputs, attention_mask, b_input_embeddings=False, b_logits=False):
        """

        :param inputs:
        :param attention_mask:
        :param input_embeddings: whether or not input embeddings are used (combination of token embeddings, segment embeddings and position embeddings)
        :param b_logits: Bool whether or not to return logits
        :return:
        """
        embeddings = self.get_output_embeddings(inputs=inputs, attention_mask=attention_mask, b_input_embeddings=b_input_embeddings)
        pred = self.classifier(embeddings, b_logits=b_logits)
        return pred

    def get_input_embeddings(self, input_ids):
        input_embeddings = self.base_model.bert.embeddings(input_ids)
        return input_embeddings

    def get_output_embeddings(self, inputs, attention_mask, b_input_embeddings=False):
        # attention_mask is neccessary, also for evaluation
        if not b_input_embeddings:
            # inputs are input_ids
            outputs = self.base_model(inputs, attention_mask=attention_mask, output_hidden_states=True)
        else:
            # inputs are input embeddings
            # TODO: Why is extended_attention_mask neccessary to achieve same results?
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # extended_attention_mask = extended_attention_mask.to(dtype=next(self.base_model.parameters()).dtype) # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            outputs = self.base_model.bert.encoder(inputs, attention_mask=extended_attention_mask, output_hidden_states=True)
        # here I use only representation of <CLS> token, but you can easily use more tokens,
        # maybe do some pooling / RNNs... go crazy here!
        embeddings = outputs['hidden_states'][12][:, 0, :]
        return embeddings


class CustomClassifier(torch.nn.Module):
    # call model.train() before training to activate dropout
    # call model.eval() before evaluation to deactivate dropout
    # for that, dropout must be definied within the constructor
    def __init__(self, hidden_layer, input_size=768, dropout=0.5):
        """Set dropout to 0 for no dropout"""
        super().__init__()

        modules = []
        in_features = input_size
        for idx, out_features in enumerate(hidden_layer):
            modules.append((f'layer{idx+1}', torch.nn.Linear(in_features=in_features, out_features=out_features)))
            modules.append((f'relu{idx+1}', torch.nn.ReLU()))
            modules.append((f'dropout{idx+1}', torch.nn.Dropout(p=dropout)))
            in_features = out_features
        modules.append((f'out', torch.nn.Linear(in_features=in_features, out_features=2)))
        modules.append((f'out_softmax', torch.nn.Softmax(dim=1)))

        self.classifier = torch.nn.Sequential(collections.OrderedDict(modules))
        # Weight initialization
        # for layer in self.classifier:
        #     if isinstance(layer, torch.nn.Linear):
        #         layer.weight.data.normal_(mean=0.0, std=0.02)
        #         if layer.bias is not None:
        #             layer.bias.data.zero_()

    def forward(self, x, b_logits=False):
        if b_logits:
            # Remove output softmax layer
            # CrossEntropyLoss() has already included a softmax layer inside
            return torch.nn.Sequential(*[self.classifier[i] for i in range(len(self.classifier)-1)])(x)
        else:
            return self.classifier(x)