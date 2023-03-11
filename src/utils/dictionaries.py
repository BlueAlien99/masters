stopwords = ['a', 'and', 'of', 'to']
proper_nouns = ['Alyokhina', 'Ankeet', 'AreoMexico', 'CENC', 'Dzhokhar', 'Equatoguineans', 'Kweifiya', 'Phailin',
                'Tsarnaev', 'aeromexico', 'anmen']
# Check if lemmatisation helps
others = ['gemmed', 'outlooking']

ignore_list = [*stopwords, *proper_nouns, *others]

uk_to_us = {
    'cancelled': 'canceled',
    'centre': 'center',
    'favourite': 'favorite',
    'fertiliser': 'fertilizer',
    'grey': 'gray',
    'honour': 'honor',
    'hospitalised': 'hospitalized',
    'legalise': 'legalize',
    'neighbours': 'neighbors',
    'offences': 'offenses',
    'paralyse': 'paralyze'
}

autocorrect = {
    'acouch': 'a couch',
    'attery': 'battery',
    'batery': 'battery',
    'bathingsuit': 'bathing suit',
    'batterty': 'battery',
    'bbulb': 'bulb',
    'becaquse': 'because',
    'bttery': 'battery',
    'circuts': 'circuits',
    'cloosed': 'closed',
    'conatined': 'contained',
    'conncted': 'connected',
    'connecton': 'connection',
    'connedted': 'connected',
    'connetced': 'connected',
    'contaied': 'contained',
    'contiained': 'contained',
    'dfferent': 'different',
    'differentclosed': 'different closed',
    'doesnt': "doesn't",
    'dressedin': 'dressed in',
    'electrial': 'electrical',
    'kitchendiner': 'kitchen diner',
    'neavtive': 'negative',
    'negavtive': 'negative',
    'negtive': 'negative',
    'papth': 'path',
    'parralel': 'parallel',
    'positie': 'positive',
    'posititve': 'positive',
    'ppositive': 'positive',
    'reacion': 'reaction',
    'rebelheld': 'rebel held',
    'separarted': 'separated',
    'separted': 'separated',
    'serperated': 'separated',
    'surfung': 'surfing',
    'swithch': 'switch',
    'terminak': 'terminal',
    'terminl': 'terminal',
    'termnal': 'terminal',
    'thebulb': 'the bulb',
    'tterminals': 'terminals',
    'whithin': 'within'
}
