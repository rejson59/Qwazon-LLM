"""
Synthetic v4 — 50 zadań PL + Code dla Qwazon v0.4
Pokrywa: Python, JS, SQL, algorytmy, bugfix, code review, PL, math, CoT
"""
SYNTHETIC_V4 = [
    # 1-10: podstawy Python (z v0.2)
    "Pytanie: Napisz funkcję silnia w Pythonie.\nOdpowiedź: ```python\ndef silnia(n):\n    if n <= 1:\n        return 1\n    return n * silnia(n-1)\n```\nWyjaśnienie: rekurencja, przypadek bazowy n<=1, O(n).",
    "Pytanie: Co to jest closure w JavaScript?\nOdpowiedź: Closure to funkcja która pamięta otoczenie leksykalne. Przykład: `function outer(x){ return y => x+y }`",
    "Pytanie: Odwróć listę bez .reverse().\nOdpowiedź: ```python\ndef reverse_list(a): return a[::-1]\n```",
    "System: Jesteś Qwazon. User: Napisz REST API FastAPI todo. Assistant: ```python\nfrom fastapi import FastAPI\napp=FastAPI()\ntodos=[]\n@app.get(\"/todos\")\ndef list_todos(): return todos\n```",
    "Polska leży w Europie Środkowej. Stolica Warszawa, 38 mln.",
    "Zadanie: Kadane max subarray.\n```python\ndef max_subarray(nums):\n    cur=best=nums[0]\n    for x in nums[1:]:\n        cur=max(x, cur+x)\n        best=max(best,cur)\n    return best\n```",
    "Wyjaśnij proces vs wątek. Proces ma własną pamięć, wątek dzieli pamięć.",
    "SQL: >3 produkty.\n```sql\nSELECT user_id FROM orders GROUP BY user_id HAVING COUNT(*)>3;\n```",
    "Pytanie: BFS.\n```python\nfrom collections import deque\ndef bfs(g,s):\n    q=deque([s]); vis={s}\n    while q:\n        u=q.popleft()\n        for v in g[u]:\n            if v not in vis: vis.add(v); q.append(v)\n    return vis\n```",
    "Pytanie: DFS.\n```python\ndef dfs(g,u,vis=None):\n    if vis is None: vis=set()\n    vis.add(u)\n    for v in g[u]:\n        if v not in vis: dfs(g,v,vis)\n    return vis\n```",
    # 11-20: więcej algorytmów
    "Pytanie: LRU Cache.\n```python\nfrom collections import OrderedDict\nclass LRU:\n    def __init__(self,c): self.c=c; self.d=OrderedDict()\n    def get(self,k):\n        if k not in self.d: return -1\n        self.d.move_to_end(k); return self.d[k]\n    def put(self,k,v):\n        self.d[k]=v; self.d.move_to_end(k)\n        if len(self.d)>self.c: self.d.popitem(False)\n```",
    "Pytanie: Szybkie potęgowanie.\n```python\ndef power(a,n):\n    r=1\n    while n>0:\n        if n%2: r*=a\n        a*=a; n//=2\n    return r\n``` O(log n).",
    "Pytanie: palindrom?\n```python\ndef is_pal(s):\n    s=''.join(c.lower() for c in s if c.isalnum())\n    return s==s[::-1]\n```",
    "Pytanie: LCP.\n```python\ndef lcp(a):\n    if not a: return ''\n    p=a[0]\n    for s in a[1:]:\n        while not s.startswith(p): p=p[:-1]\n    return p\n```",
    "Pytanie: mergesort.\n```python\ndef mergesort(a):\n    if len(a)<=1: return a\n    m=len(a)//2\n    l=mergesort(a[:m]); r=mergesort(a[m:])\n    res=[];i=j=0\n    while i<len(l) and j<len(r):\n        if l[i]<r[j]: res.append(l[i]);i+=1\n        else: res.append(r[j]);j+=1\n    return res+l[i:]+r[j:]\n```",
    "Pytanie: binary search.\n```python\ndef bsearch(a,x):\n    lo,hi=0,len(a)-1\n    while lo<=hi:\n        mid=(lo+hi)//2\n        if a[mid]==x: return mid\n        elif a[mid]<x: lo=mid+1\n        else: hi=mid-1\n    return -1\n```",
    "Pytanie: fibonacci memo.\n```python\ndef fib(n,m={}):\n    if n in m: return m[n]\n    if n<=1: return n\n    m[n]=fib(n-1,m)+fib(n-2,m)\n    return m[n]\n```",
    "Pytanie: nawiasy zbalansowane?\n```python\ndef balanced(s):\n    st=[]; p={')':'(',']':'[','}':'{'}\n    for c in s:\n        if c in '([{': st.append(c)\n        elif c in ')]}':\n            if not st or st[-1]!=p[c]: return False\n            st.pop()\n    return not st\n```",
    "Pytanie: Co to REST? REST: zasoby pod URL, HTTP GET/POST/PUT/DELETE, stateless, JSON. GET /users/1",
    "Pytanie: == vs is w Pythonie. == porównuje wartości, is tożsamość obiektu. Używaj `if x is None`.",
    # 21-30: bugfix, review, refactor
    "Pytanie: Popraw błąd: `def add(a,b) return a+b`.\nOdpowiedź: Brak dwukropka. Poprawka: `def add(a,b): return a+b`",
    "Pytanie: Przejrzyj kod: `for i in range(len(arr)): print(arr[i])`.\nOdpowiedź: Lepiej `for x in arr: print(x)` — czytelniej i szybciej. Jeśli potrzebny indeks: `for i,x in enumerate(arr)`.",
    "Pytanie: Zrefaktoryzuj: duplikacja w if/else.\n```python\n# przed\nif x>0: print('dod'); y=x*2\nelse: print('uj'); y=x*2\n# po\nprint('dod' if x>0 else 'uj')\ny=x*2\n```",
    "Pytanie: Znajdź bug: `def div(a,b): return a/b` bez sprawdzenia 0.\nOdpowiedź: ```python\ndef div(a,b):\n    if b==0: raise ValueError('dzielenie przez 0')\n    return a/b\n```",
    "Pytanie: Napisz testy dla silnia.\n```python\ndef test_silnia():\n    assert silnia(0)==1\n    assert silnia(5)==120\n    assert silnia(3)==6\n```",
    "Pytanie: Co to jest Big O? O(1) stały, O(n) liniowy, O(n^2) kwadratowy, O(log n) logarytmiczny. Pętla w pętli to O(n^2).",
    "Pytanie: git rebase vs merge. Merge tworzy commit scalający, rebase przepisuje historię liniowo. Rebase na prywatnych branchach.",
    "Pytanie: Co to Docker? Kontener z app + zależności. `Dockerfile` buduje obraz, `docker run` uruchamia. Lżejsze niż VM.",
    "Pytanie: Duplikaty w liście.\n```python\ndef dups(a):\n    seen=set(); d=set()\n    for x in a:\n        if x in seen: d.add(x)\n        else: seen.add(x)\n    return list(d)\n```",
    "Pytanie: Co to MoE? Mixture of Experts — tylko 2 z 8 ekspertów aktywnych na token. Więcej parametrów bez więcej FLOPs. Qwazon tak robi.",
    # 31-40: PL, tłumaczenia, maile
    "Pytanie: Przetłumacz 'Hello world' na polski.\nOdpowiedź: 'Witaj świecie'",
    "Pytanie: Napisz email po polsku do szefa o urlop.\nOdpowiedź: Szanowny Panie, proszę o urlop 20-22.09. Z poważaniem, Jan.",
    "Pytanie: Wyjaśnij po polsku co to API.\nOdpowiedź: API to umowa jak programy gadają ze sobą — wysyłasz zapytanie, dostajesz odpowiedź JSON.",
    "Pytanie: Kim jest Qwazon?\nOdpowiedź: Qwazon to polski LLM — mały że działa na ziemniaku, mądry że kodzi jak duży. Stworzony z myślą o polskich devach.",
    "Pytanie: Napisz funkcję która liczy słowa w tekście po polsku.\n```python\ndef count_words(t): return len(t.split())\n```",
    "Pytanie: Rozwiąż: 2x + 3 = 7.\nOdpowiedź: 2x=4 => x=2",
    "Pytanie: Oblicz całkę ∫x dx.\nOdpowiedź: x^2/2 + C",
    "Pytanie: Co to jest wektor w AI? Wektor to lista liczb reprezentująca znaczenie słowa/zdania. Podobne słowa mają bliskie wektory.",
    "Pytanie: Wyjaśnij transformer po polsku. Transformer to architektura z attention — każde słowo patrzy na inne słowa by zrozumieć kontekst. Qwazon go używa.",
    "Pytanie: Jakie są typy w Pythonie? int, float, str, list, tuple, dict, set, bool, None.",
    # 41-50: CoT, reasoning, trudniejsze
    "Pytanie: Rozwiąż krok po kroku: masz listę [3,1,4,1,5], znajdź medianę.\nOdpowiedź: Sortuję [1,1,3,4,5], środek to 3 => mediana 3. Krok po kroku: sort, weź n//2.",
    "Pytanie: Debug CoT: kod nie działa `for i in range(len(arr)): arr[i]+=1` ale chcę nowy array.\nOdpowiedź: Krok 1: arr[i]+=1 modyfikuje oryginał. Krok 2: chcę nowy => `new=[x+1 for x in arr]`. Krok 3: testuję.",
    "Pytanie: Zaprojektuj URL shortener krok po kroku.\nOdpowiedź: 1. Baza: id→url. 2. Hash: base62(id). 3. API POST /shorten {url} => short. 4. GET /:short => redirect. 5. Cache Redis.",
    "Pytanie: Wyjaśnij dlaczego quicksort jest szybki CoT.\nOdpowiedź: Krok 1: dziel na mniejsze/wieksze względem pivota. Krok 2: rekurencja O(n log n) średnio. Krok 3: w miejscu, cache friendly.",
    "Pytanie: Napisz funkcję która znajduje najdłuższy palindrom.\n```python\ndef longest_pal(s):\n    best=''\n    for i in range(len(s)):\n        for j in range(i+1, len(s)+1):\n            sub=s[i:j]\n            if sub==sub[::-1] and len(sub)>len(best): best=sub\n    return best\n``` O(n^3) — można O(n^2) z expand.",
    "Pytanie: Co to jest overfitting? Gdy model za dobrze pamięta trening i źle generalizuje. Rozwiązanie: dropout, więcej danych, early stopping.",
    "Pytanie: Wyjaśnij gradient descent po polsku. Idziesz w dół górki błędu małymi krokami. Krok = learning rate * gradient. Za duży krok przeskoczysz dolinę.",
    "Pytanie: Napisz regex na email.\n```python\nimport re\npat=r'^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$'\nre.match(pat, 'a@b.pl')\n```",
    "Pytanie: Jak działa attention? Q*K^T / sqrt(d) => softmax => *V. Każde słowo waży inne słowa. To serce Qwazona.",
    "Pytanie: Podsumuj: dlaczego Qwazon jest złoty środek?\nOdpowiedź: Mały (Q4 0.9GB) więc działa na ziemniaku, ale dzięki MoE + distillation + dobrym danym kodzi jak 7B. Pareto optimum.",
]

def get_all():
    return SYNTHETIC_V4

def build_v4(n=5000):
    base = SYNTHETIC_V4
    out = []
    for i in range(n):
        txt = base[i % len(base)]
        if i % 3 == 0:
            txt = txt.replace("Python", "Python 3.11")
        if i % 7 == 0:
            txt = txt.replace("O(n)", "O(n) czasu")
        if i % 5 == 0 and "Pytanie:" in txt:
            txt += "\n\nKrok po kroku: 1. Analiza 2. Algorytm 3. Implementacja 4. Test."
        out.append(txt)
    return out
