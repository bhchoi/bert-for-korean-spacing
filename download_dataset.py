from Korpora import Korpora

# namuwikitext = Korpora.load('namuwikitext')

# or
# Korpora.fetch('namuwikitext')

namuwikitext = Korpora.load('kcbert')
Korpora.fetch('kcbert', root_dir='./data')