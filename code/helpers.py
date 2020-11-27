import os, re, datetime

# to be expanded as needed in the future
# wizard.media@virgin.net is a common media contact for wiley.com (e.g. https://www.eurekalert.org/pub_releases/2009-06/w-bic060909.php)
KNOWN_JOURNAL_LIST = '''
cell.com
wiley.com
bmj.com
plos.org
elsevier.com
bmjgroup.com
wolterskluwer.com
biomedcentral.com
oup.com
liebertpub.com
lancet.com
springer.com
the-jci.org
sagepub.com
wiley-vch.de
benthamscience.net
benthamscience.org
iospress.com
virgin.net
'''.strip().split()

# add to the following list those non-canonical domain names
# in the future, we should compile a comprehensive list
KNOWN_ACADEMIC_LIST = 'epfl.ch ethz.ch kuleuven.be'.split()
KNOWN_RESEARCH_LIST = 'cnrs.fr cnic.es cnio.es riken.jp sissa.it crg.eu mpg.de dzne.de dkfz.de helmholtz-hzi.de gwdg.de dzd-ev.de ufz.de'.split()

def get_or_create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def get_type_of_eureka_source(x):
    contact = x['contact']
    institution = x['institution']

    if re.search(r'(\.gov|\.mil|\bgov\.\w+|\bmil\.\w+)$', contact):
        return 'Government'

    if contact in KNOWN_JOURNAL_LIST or re.search(r'(Publish|Press|Journal|Publication|Lancet|PLOS)', institution, flags=re.IGNORECASE):
        return 'Journal'

    if contact in KNOWN_ACADEMIC_LIST or re.search(r'(\.edu|\bedu\.\w+|\bac\.\w+|\buni-.*?\.de|tcd\.ie)$', contact) or re.search(r'(Universit|Institut)', institution):
        return 'University'

    if re.search(r'(\.org|\borg\.\w+)$', contact) or re.search(r'(Association|Fundaz|Foundation|Society|Organi.ation|Academ)', institution, flags=re.IGNORECASE):
        return 'Academic society'

    if re.search(r'(gmail.com|mac.com|hotmail.com|yahoo.com|aol.com)$', contact):
        return 'Email'
    else:
        return 'Others'

def get_current_date_str():
    return datetime.date.today().strftime("%Y%m%d")

def display_first(df, title=None):
    print()
    if title is None:
        print('------------')
    else:
        print(f'--- {title} ---')
    if len(df)<1:
        raise ValueError('\n********************\n- !!!!!!! ERROR : display_first: empty df\n=====================\n\n')
        sys.exit()
    print(df.iloc[0])
    print(f'--- size: {len(df)} ---')

def display_df_cnt(df, attr, n=5, sort_by=None):
    n_ = 5 if n==-1 else n
    print(f'\n== value counts of {attr} (for top {n_} only) ==')
    print(df[attr].value_counts()[:n_])
    print(f'-- size of df: {len(df)} / uniques: {len(df[attr].unique())} --')
    if n==-1:
        tmp = df[attr].value_counts().reset_index()
        tmp.columns = [attr, 'cnt']
        if sort_by is not None:
            print(f'\n== sort by {sort_by} ==')
            print(tmp.sort_values(sort_by))
        print(f'-- count of rows with at least 2 same "{attr}": {len(tmp[tmp["cnt"]>1])}')
        return tmp
    print()
