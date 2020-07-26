import io
import sys
import bz2

import enchant
import Levenshtein as ld

end = enchant.Dict('en')

def get_edits(l1, l2):
    idx = 0x3400
    mapp = {}
    rmapp = {}
    for w in l1.split()+l2.split():
        if w in mapp:
            continue
        mapp[w] = chr(idx)
        rmapp[chr(idx)] = w
        idx += 1
    l1 = ''.join([mapp[w] for w in l1.split()])
    l2 = ''.join([mapp[w] for w in l2.split()])
    edits = ld.editops(l1, l2)
    uedits = []
    ruedits = []
    ied, ded, red = 0, 0, 0
    for edit in edits:
        edit, id1, id2 = edit
        if edit == 'delete':
            ded += 1
            uedits.append('<delete><%s><%d>' %(rmapp[l1[id1]], id1+1))
        elif edit == 'insert':
            ied += 1
            uedits.append('<insert><%s><%d>' %(rmapp[l2[id2]], id1+1))
        elif edit == 'replace':
            red += 1
            ruedits.append((rmapp[l1[id1]], rmapp[l2[id2]]))
            uedits.append('<replace><%s><%d><%s><%d>' %(rmapp[l1[id1]], id1+1, rmapp[l2[id2]], id2+1))
    typo = False
    if ied == ded == 0:
        for edit in ruedits:
            if (not end.check(edit[0])) and end.check(edit[1]):
                typo = True
                break
    return '|||'.join(uedits), typo
    
with bz2.open(sys.argv[1]) as fp:
    group = []
    for line in fp:
        title, revision, src, tgt = line.decode('utf-8').strip().split('\t')
        if revision.startswith('Begin'):
            if group:
                first = True
                for l1,l2 in zip(group[:-1], group[1:]):
                    edits, typo = get_edits(l1[-1], l2[-1])
                    if typo:
                        print('\t'.join(l1+[l2[2]]+[edits]), file=sys.stderr)
                    else:
                        if first:
                            first = False
                            print('\t'.join([title]+["Begin_Revisions"]+[l1[2]]+[l2[2]]+[edits]))
                        else:
                            print('\t'.join([title]+["Inside_Revisions"]+[l1[2]]+[l2[2]]+[edits]))
            group = [[title, revision, src], [title, revision, tgt]]
        else:
            group.append([title, revision, tgt])
