from pathlib import Path
import sys, importlib.metadata
base=Path(r'D:\Projects\Python\Light-FE-KFormer-CR')
env=base/'environment'/'.venv312'
from pip._vendor.distlib.scripts import ScriptMaker
maker=ScriptMaker(None,str(env/'Scripts'))
maker.executable=str(env/'Scripts'/'python.exe')
maker.clobber=True
maker.variants={''}
count=0
for dist in importlib.metadata.distributions(path=[str(env/'Lib'/'site-packages')]):
    for entry in dist.entry_points:
        if entry.group in ('console_scripts','gui_scripts'):
            maker.make(f'{entry.name} = {entry.value}',options={'gui':entry.group=='gui_scripts'})
            count+=1
cfg=env/'pyvenv.cfg'
cfg.write_text(cfg.read_text().replace(str(base/'result'/'.venv312'),str(env)))
project=base/'Light-FE-KFormer-CR'
p=project/'make_report.py'
s=p.read_text(encoding='utf-8-sig').replace('../result/.venv312','../environment/.venv312').replace('精确版本见 requirements-lock.txt','精确版本见 ../environment/requirements-lock.txt').replace('Python 3.12 独立环境在 .venv312','Python 3.12 独立环境在 ../environment/.venv312').replace('早期失败日志保留于 logs，','早期失败日志已清理，').replace('- .venv、.venv312、tools、tmp、cache：环境尝试、实际环境、工具及中间文件，均保留在 result 内。','- ../environment/.venv312：有效运行环境；旧环境和安装缓存已清理。运行缓存仍在 result/tmp 和 result/cache。')
p.write_text(s,encoding='utf-8')
# Set explicit interpreter paths for saved run configurations, avoiding stale SDK entries.
import xml.etree.ElementTree as ET
for folder in [base/'.idea',project/'.idea']:
    p=folder/'workspace.xml'
    if p.exists():
        tree=ET.parse(p)
        for config in tree.findall(".//configuration[@type='PythonConfigurationType']"):
            for key,value in [('SDK_HOME',str(env/'Scripts'/'python.exe')),('IS_MODULE_SDK','false')]:
                opt=config.find(f"option[@name='{key}']")
                if opt is None: opt=ET.SubElement(config,'option',{'name':key})
                opt.set('value',value)
        tree.write(p,encoding='utf-8',xml_declaration=True)
print('Regenerated launchers:',count)
