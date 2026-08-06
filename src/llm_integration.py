import os
import requests
from typing import Dict, Optional, Generator
from dotenv import load_dotenv
import logging
from openai import OpenAI
load_dotenv()

class PatentGenerator:
    """
    专利生成器类，用于生成和修订专利文档。
    """
    def __init__(self):
        """
        初始化专利生成器。
        """
        self.api_base = os.getenv('LLM_API_BASE')
        self.api_key = os.getenv('LLM_API_KEY')
        self.model = os.getenv('LLM_MODEL')
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.current_draft: Dict[str, str] = {}
        self.query: str = ""
        self.context: str = ""
        self.glossary: str = ""  # 【阶段1】术语表：从技术文档抽取的关键术语，贯穿说明书/摘要/权利要求生成
        # 【消融开关】stage 档位控制启用哪些改进，用于消融实验（同一套代码跑不同阶段）：
        #   0 = 基线(无术语表, temp0.5/0.6, 独立生成)
        #   1 = +术语约束(术语表 + temp0.2)
        #   2 = +分层串联(在1的基础上, 权利要求/摘要依赖已生成的说明书)
        self.stage: int = 2

    def build_glossary(self, tech_doc: str) -> str:
        """
        【阶段1新增】从技术文档中抽取关键技术术语，构建术语表。
        术语表用于约束后续生成，保证同一部件/概念在各部分中用词一致（解决术语漂移）。

        参数:
        tech_doc (str): 技术文档（交底书）内容。

        返回:
        str: 术语表文本，每行一个术语。
        """
        prompt = f'''你是专利术语抽取助手。请从下面的技术文档中，抽取所有关键技术术语（包括：部件/模块名称、技术方法名称、专有名词、关键参数名）。

### 要求：
1. 只输出术语本身，每行一个，不要编号、不要解释。
2. 保持术语的原始写法，不要改写或翻译。
3. 优先抽取会在权利要求中作为技术特征出现的名词性术语。

### 技术文档：
{tech_doc}

请输出术语列表：'''
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # 抽取任务要确定性，温度设0
                stream=False,
            )
            self.glossary = response.choices[0].message.content.strip()
            return self.glossary
        except Exception as e:
            logging.error(f'术语表构建失败: {str(e)}')
            self.glossary = ""
            return ""

    def generate_draft(self, query: str, context: str, doc_type: str) -> str:
        """
        根据给定的查询、上下文和文档类型生成专利文档初稿。

        参数:
        query (str): 相关专利内容，用于生成文档。
        context (str): 参考技术文档，用于模仿风格和格式。
        doc_type (str): 文档类型，如 "说明书", "摘要", "权利要求书"。

        返回:
        str: 生成的专利文档初稿内容。
        """
        self.query = query
        self.context = context

        # 【阶段1】术语约束段：若已构建术语表，则强制生成时沿用这些术语（stage>=1 才启用）
        glossary_section = ""
        if self.stage >= 1 and self.glossary:
            glossary_section = f'''
### 术语表（必须严格遵守）：
以下是本发明的关键术语。生成时，凡涉及这些概念，必须逐字使用下列术语，禁止替换为同义词、简称或自行改写；也禁止发明术语表和技术文档中都不存在的新部件名称。
{self.glossary}
'''

        # 【阶段2】串联依赖段：生成权利要求/摘要时，注入已生成的说明书作为唯一事实依据（stage>=2 才启用）
        # 说明书信息最全，先生成；权利要求和摘要必须基于它，禁止引入说明书没有的特征
        upstream_section = ""
        if self.stage >= 2 and doc_type in ("权利要求书", "权利要求") and "说明书" in self.current_draft:
            upstream_section = f'''
### 已生成的说明书（权利要求的唯一事实依据，必须严格遵守）：
以下是本发明已经定稿的说明书。请注意：
1. 权利要求中出现的每一个技术特征，都必须能在下面的说明书中找到对应描述（专利法要求权利要求得到说明书支撑）。
2. 禁止引入说明书中未描述的部件、模块或技术特征。
3. 术语必须与说明书完全一致。
{self.current_draft["说明书"]}
'''
        elif self.stage >= 2 and doc_type == "摘要" and "说明书" in self.current_draft:
            upstream_section = f'''
### 已生成的说明书（摘要须据此浓缩，必须严格遵守）：
摘要是下面这份说明书的高度概括。请只概括说明书中已有的技术方案，不要引入说明书没有的内容，术语与说明书保持一致。
{self.current_draft["说明书"]}
'''

        # 构建提示信息
        prompt = f'''你是一个专业的专利申请文档撰写助手。请参考以下技术文档的行文风格和格式，并基于提供的相关专利内容，生成一段高质量的{doc_type}。该内容应直接适用于专利申请书中的对应部分。
{glossary_section}
{upstream_section}
### 要求：
1. **风格和格式**：严格模仿【参考技术文档】的行文风格、段落结构和术语使用。
2. **格式要求**：输出使用XML标签包裹，如果有标题，则使用<标题> </标题> 将标题内容包裹，如果有段落，则使用<段落></段落>将段落内容包裹，结构示例：
<标题>标题内容</标题>
<段落>段落内容...</段落>
每个段落必须用<段落>标签包裹，标题用<标题>标签。
3. **内容生成**：基于【相关专利内容】生成具体的技术描述，确保逻辑清晰、表达准确，不要编造相关专利内容中不存在的数据。
4. **专业性**：使用专利申请中常见的专业术语和表达方式。
5. **完整性**：确保生成的{doc_type}内容完整，包含所有必要的技术细节和描述。
6. **术语一致性**：全文对同一部件或概念必须使用完全一致的名称，不得在不同段落使用不同说法。
7. **专利撰写规范**：
   - 如果{doc_type}是**权利要求**，请确保准确描述发明的技术特征和保护范围。
   - 如果{doc_type}是**说明书**，请详细描述技术方案的实施方式、优点和具体示例。
   - 如果{doc_type}是**摘要**，则不需要输出 <标题> </标题>，仅仅输出段落即可。
### 参考技术文档（请模仿其风格和格式）：
{context}

### 相关专利内容（请基于此内容生成）：
{query}

请根据以上要求，生成专业的{doc_type}。'''

        try:
            # 【消融】stage0 用原版高温 0.5，stage>=1 降到 0.2 减少换词
            gen_temp = 0.5 if self.stage == 0 else 0.2
            # 调用 OpenAI API 生成文档
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                temperature=gen_temp,  # 【消融】stage0=0.5 / stage>=1=0.2
                max_tokens=8000,       # 避免长说明书被默认上限截断
                stream=False,
            )
            
            content = response.choices[0].message.content
            self.current_draft[doc_type] = content
            return content
        except Exception as e:
            error_msg = f'生成{doc_type}失败: {str(e)}'
            logging.error(error_msg)
            return error_msg

    def revise_draft(self, feedback: str, doc_type: str) -> str:
        """
        根据用户反馈修订专利文档初稿。

        参数:
        feedback (str): 用户的修改意见。
        doc_type (str): 文档类型，如 "说明书", "摘要", "权利要求书"。

        返回:
        str: 修订后的专利文档内容。
        """
        if doc_type not in self.current_draft:
            return '请先生成初稿'

        # 【阶段1】修订时同样注入术语约束（stage>=1 才启用）
        glossary_section = ""
        if self.stage >= 1 and self.glossary:
            glossary_section = f'''
### 术语表（必须严格遵守）：
以下术语必须逐字沿用，禁止替换为同义词、简称或新造名称：
{self.glossary}
'''

        # 【阶段4】重写时同样注入说明书作事实依据（stage>=2 且重写权利要求/摘要时）
        # 闭环重写顺序为 说明书→权利要求→摘要，故下游读到的是本轮已修正的说明书
        upstream_section = ""
        if self.stage >= 2 and doc_type in ("权利要求书", "权利要求") and "说明书" in self.current_draft:
            upstream_section = f'''
### 已生成的说明书（权利要求的唯一事实依据，必须严格遵守）：
以下是本发明已定稿的说明书。权利要求中的每个技术特征都必须能在说明书中找到对应描述，禁止引入说明书未描述的部件、模块或技术特征，术语须与说明书完全一致。
{self.current_draft["说明书"]}
'''
        elif self.stage >= 2 and doc_type == "摘要" and "说明书" in self.current_draft:
            upstream_section = f'''
### 已生成的说明书（摘要须据此浓缩，必须严格遵守）：
摘要是下面这份说明书的高度概括。请只概括说明书中已有的技术方案，不要引入说明书没有的内容，术语与说明书保持一致。
{self.current_draft["说明书"]}
'''

        # 构建修订提示信息
        prompt = f'''你是一个专业的专利申请文档撰写助手。请根据用户反馈修改{doc_type}，同时确保修订后的内容仍然符合原始技术文档的风格和格式，并基于相关专利内容。
{glossary_section}
{upstream_section}
### 要求：
1. **风格和格式**：继续模仿【原始技术文档】的行文风格和格式。
2. **内容修改**：根据【修改意见】对【当前版本】进行修订，确保修改后的内容准确、完整。
3. **专业性**：保持专利申请的专业术语和表达方式。
4. **一致性**：确保修订后的内容与【相关专利内容】保持一致，且对同一部件/概念全文用词一致。
5. **专利撰写规范**：
   - 如果{doc_type}是**权利要求**，请确保技术特征和保护范围的准确性。
   - 如果{doc_type}是**说明书**部分，请确保技术方案的描述详细且具有可实施性。

### 模仿和参考的原始技术文档（参考风格和格式）：
{self.context}

### 相关专利内容：
{self.query}

### 当前版本：
{self.current_draft[doc_type]}

### 修改意见：
{feedback}

请根据以上要求，生成修订后的{doc_type}。'''

        try:
            # 【消融】stage0 用原版高温 0.6，stage>=1 降到 0.2
            rev_temp = 0.6 if self.stage == 0 else 0.2
            # 调用 OpenAI API 修订文档
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                temperature=rev_temp,  # 【消融】stage0=0.6 / stage>=1=0.2
                max_tokens=8000,       # 避免长说明书被默认上限截断
                stream=False,
            )
            
            content = response.choices[0].message.content
            self.current_draft[doc_type] = content
            return content
        except Exception as e:
            error_msg = f'修订{doc_type}失败: {str(e)}'
            logging.error(error_msg)
            return error_msg
