import spacy


nlp = spacy.load("ru_core_news_md")
doc = nlp("Опыт программирования в 1С от двух лет; Знание одной или нескольких типовых конфигураций 1С (желательно УТ); Знание языка запросов, СКД, управляемых форм")
for ent in doc.ents:
    print(ent.text, ent.label_)

from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,

    PER,
    NamesExtractor,

    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)
text = "Опыт программирования в 1С от двух лет; Знание одной или нескольких типовых конфигураций 1С (желательно УТ); Знание языка запросов, СКД, управляемых форм"
doc = Doc(text)
doc.segment(segmenter)
doc.tag_ner(ner_tagger)

for span in doc.spans:
    if span.type == PER:
        print("111")
        print(span.extract_fact(names_extractor))




print(doc)