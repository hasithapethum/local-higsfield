import json,urllib.request,sys
def submit(api,run=False):
    """POST then, if it's only a validation test, remove from queue."""
    req=urllib.request.Request('http://127.0.0.1:8188/prompt',data=json.dumps({'prompt':api,'client_id':'h3studio-test'}).encode(),headers={'Content-Type':'application/json'})
    try:
        r=json.loads(urllib.request.urlopen(req).read())
        if r.get('node_errors'):
            print('PARTIAL errors:')
            for k,v in r['node_errors'].items(): print(' ',k,v['class_type'],[x['message']+' | '+x['details'] for x in v['errors']])
        else: print('valid')
        if not run:
            urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8188/queue',data=json.dumps({'delete':[r['prompt_id']]}).encode(),headers={'Content-Type':'application/json'}))
        return r
    except urllib.error.HTTPError as e:
        r=json.loads(e.read()); print(r['error']['message'])
        for k,v in r.get('node_errors',{}).items(): print(' ',k,v['class_type'],[x['message']+' | '+x['details'] for x in v['errors']])
if __name__=='__main__':
    import wf_h3swap as w
    submit(w.build({'video':'a.mp4','character':'Krea2_turbo_00006_.png'}).to_api())
