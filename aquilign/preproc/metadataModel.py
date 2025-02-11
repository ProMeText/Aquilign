import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoModelForTokenClassification, BertForTokenClassification, PreTrainedModel


# class BertWithMetadata(nn.Module):
#     def __init__(self, pretrained_model_name, num_metadata_features, num_classes):
#         super(BertWithMetadata, self).__init__(config)
class BertWithMetadata(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.freeze_metadata = config.freeze_metadata
        self.num_classes = config.num_labels
        num_metadata_features = config.num_metadata_features
        pretrained_model_name = config.name_or_path
        self.bert = BertForTokenClassification.from_pretrained(pretrained_model_name, num_labels=self.num_classes)
        hidden_size = self.bert.config.hidden_size
        # Définir une couche d'embedding pour les métadonnées
        self.metadata_embedding = nn.Embedding(num_metadata_features, hidden_size)
        # Couche de classification
        self.classifier = nn.Linear(hidden_size, self.num_classes)


    def forward(self, input_ids, attention_mask, metadata, labels=None):
        # Passer les entrées dans BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Extraire la dernière couche du modèle BERT
        cls_output = bert_output.hidden_states[-1]
        
        # Passer les métadonnées à travers leur couche d'embedding
        if self.freeze_metadata:
            logits = self.classifier(cls_output)
        else:
            metadata_embed = self.metadata_embedding(metadata)
            # Fusionner la sortie de BERT et l'embed des métadonnées
            combined_output = cls_output + metadata_embed  # Fusionner par addition
            # Classifier
            
            logits = self.classifier(combined_output)

        # If labels are provided, compute the loss
    
        loss = None
        if labels is not None:
            # Calculate loss (CrossEntropyLoss for token classification)
            loss_fct = nn.CrossEntropyLoss()
            # Flatten the logits and labels for cross entropy loss
            flattened_logits = logits.view(-1, self.num_classes)
            flattened_labels = labels.view(-1)
            loss = loss_fct(flattened_logits, flattened_labels)

        return (loss, logits) if loss is not None else logits
    
        


if __name__ == '__main__':
    
    tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
    
    texte = "Este texto habla de actualidades políticas."
    metadonnees = torch.tensor([2])  # Représentation de la catégorie
    
    inputs = tokenizer(texte, return_tensors='pt', padding=True, truncation=True)
    
    model = BertWithMetadata(pretrained_model_name='dccuchile/bert-base-spanish-wwm-cased', num_metadata_features=3, num_classes=3)
    
    logits = model(input_ids=inputs['input_ids'],
                   attention_mask=inputs['attention_mask'],
                   metadata=metadonnees)
    tokens_ids = inputs['input_ids'].tolist()
    
    print([tokenizer.convert_ids_to_tokens(ident) for ident in tokens_ids])
    print(logits)