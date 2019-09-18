import codecs
import os
from collections import defaultdict
import pickle

import pandas as pd
import click


# Process into Sequence by user
def id_generator():
    n = 0
    while True:
        yield n
        n += 1


@click.command()
@click.option('--file', default='skill_builder_data_corrected', help='File name without extension')
@click.option('-csid', '--col-sid', 'csid', default='skill_id', help='Column name of skill id')
@click.option('-cusr', '--col-usr', 'cusr', default='user_id', help='Column name of user id')
@click.option('-cans', '--col-ans', 'cans', default='correct', help='Column name of answer')
@click.option('--sort-by', default='', help='Column name to sort by')
def main(file, csid, cusr, cans, sort_by):
    run(file, csid, cusr, cans, sort_by)
    
    
def run(file, csid, cusr, cans, sort_by):
    dirname = os.path.dirname(__file__)
    infname = os.path.join(dirname, f'raw_input/{file}.csv')
    outfname = os.path.join(dirname, f'input/{file}.pickle')

    with codecs.open(infname, 'r', 'utf-8', 'ignore') as f:
        df = pd.read_csv(f)
        
    if sort_by:
        df = df.sort_values(by=sort_by)

    it = iter(id_generator())

    processed = defaultdict(list)
    problems = defaultdict(lambda: next(it))
    for idx, row in df.iterrows():
        # nanは無視する
        sid = row[csid]
        usr = row[cusr]
        ans = row[cans]
        
        if pd.isnull(sid) or pd.isnull(ans) or pd.isnull(usr):
            continue
        if float(ans) not in {0., 1.}:
            continue
        # processed[row.user_id].append((problems[row.problem_id], row.correct))
        processed[usr].append((problems[sid], int(ans)))

    print('Problems:', len(problems))
    print('Students:', len(processed))

    # Save processed data
    with open(outfname, 'wb') as f:
        pickle.dump(processed, f)


if __name__ == '__main__':
    main()