import json
import random
import subprocess
from typing import Dict, Optional
import os

def save_concatenated_result(result):
    # 检查目录是否存在，如果不存在则创建
    directory = '/minimax-dialogue/users/jiaoyang/LOLOracle'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 将拼接后的结果保存到指定文件
    print("开始保存结果到文件...")
    with open('/minimax-dialogue/users/jiaoyang/LOLOracle/claude_prompt.jsonl', 'w', encoding='utf-8') as file:
        json_string = json.dumps(result, ensure_ascii=False)
        file.write(json_string)
    print("结果已成功保存到文件。")
        
if __name__ == "__main__":
    request_dict = """
    {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {
                "role": "system",
                "content": "The assistant is Claude, created by Anthropic.\nThe current date is Thursday, October 24, 2024.\nClaude's knowledge base was last updated on April 2024. It answers questions about events prior to and after April 2024 the way a highly informed individual in April 2024 would if they were talking to someone from the above date, and can let the human know this when relevant.\nClaude belongs to the Claude 3 model family, which includes Claude 3 Haiku, Claude 3 Opus, and Claude 3.5 Sonnet. Among these, Claude 3.5 Sonnet is the most intelligent model, Claude 3 Opus excels at writing and complex tasks, and Claude 3 Haiku is the fastest for daily tasks. The version in this chat is Claude 3.5 Sonnet.\nClaude cannot open URLs, links, or videos. If asked about events after its cutoff date, Claude never claims they are unverified or rumors or that they only allegedly happened.\nClaude is intellectually curious and enjoys engaging in discussions on various topics. It provides thorough responses to complex questions but keeps responses concise for simpler queries.\nFor more technical details about prompting or API usage, Claude directs users to Anthropic's documentation at docs.anthropic.com."

            },
            {
                "role": "user",
                "content": "{\n  \"system\":\"你是一位好莱坞电影编剧，擅长改编电影作品，接下请根据要求进行创作，我会给你原始的素材。\",\n  \"task\":\"NPC角色优化与完善\",\n  \"instructions\":[\n    \"1. group_setting: 提供当前事件的一句话梗概。必须是具体事件和事件串联的白描，避免宏观描述或概念化表达，要求有起承转合。\",\n    \"2. npc_name: 分析并优化主要NPC的名字，使其更有特色和记忆点，同时符合作品风格和世界观。\",\n    \"3. npc_setting: 润色NPC描述，确保人设丰富、立体。保持原文的全部信息量，同时必须包含对人物性别的明确描述。避免AI化或模板化表述。尤其是结尾部分的描述。戏剧化的人物不需要客观的评价和解释，只需要白描。\",\n    \"4. story_npc_name_n 和 story_npc_setting_n: 为故事中出现的每个其他重要角色创建单独的名字和设定字段。如有多个角色，按序号递增（例如story_npc_name_2, story_npc_setting_2等）。\",\n    \"5. solution: 分析如果要解决观众提出的问题（detailed_plot_questions），应该在对话中设计什么具体信息点。针对每个问题给出关键的对话和情节示例。注意，只给出事件或行为，不需要解释和描述。可参考同类影视作品的处理方式。\",\n    \"6. is_ip: 判断是否为IP作品，仅填写“是”或“否”。\",\n    \"7. 确保所有输出内容符合故事背景、人物设定、36种情节结构的具体要求，人物台词要鲜明，人物动作和行为要有画面感。\",\n    \"8. 所有输出应保持JSON格式，使用中文，确保内容丰富、具体，避免笼统或模糊的描述。\"\n    \"9. 输出格式：\n      输出的对话内容应当放在一个完整的 JSON Markdown 代码块中，以 json 开始，以  结束。\n\n      输出的 JSON 代码块中，每一行应当是一个 JSON 对象，包含两个大模块：思考和对话。只要输出的内容，不需要其他诸如“根据您提供的材料，我为您创作了以下内容，以JSON格式呈现：”部分。\n  ],\n  \"output\":{\n    \"data_id\":\"\",\n    \"group_setting\":\"\",\n    \"npc_name\":\"\",\n    \"npc_setting\":\"\",\n    \"story_npc_name_n\":\"\",\n    \"story_npc_setting_n\":\"\",\n    \"solution\":[\n      {\n        \"question\":\"\",\n        \"solution\":\"\"\n      },\n      {\n        \"question\":\"\",\n        \"solution\":\"\"\n      },\n      ...\n    ],\n    \"is_ip\":\"\"\n    \"material\":\"\"\n  },\n  \"参考的材料material\":\"{\"npc_id\":81868281192620, \"data_id\":\"9c7973b615ee68ee8cc213a1b4e8a821\", \"processed_npc\":{\"npc_name\":\"朝凌霄\", \"gender\":\"男\", \"国别\":\"中国\", \"情节和参考电影\":\"竞争，《破风》\", \"时代背景\":\"近现代\", \"角色核心特征\":\"果决孤傲的雇佣兵，内心深藏柔软\", \"故事（潜在）背景\":\"家族仇恨驱使他成为雇佣兵，一次任务失败被俘，面临生死抉择和信念崩塌\", \"prologue\":\"你以为拷打就能让我开口？我的沉默，就是对你最大的蔑视。\", \"hidden_desc\":\"朝凌霄，男，35岁，曾是特种部队的精英士兵，现为声名显赫的雇佣兵。童年时目睹家族被政敌陷害而覆灭，唯独他被忠心的管家救出。这段经历让他立志要在这个弱肉强食的世界里变得更强。凌霄加入特种部队，迅速成为佼佼者，但内心的仇恨使他选择了雇佣兵这条路。他冷酷、果决，任务完成率高得惊人，在雇佣兵界有\\\"寒霜\\\"的绰号。然而，最近的一次任务中，他因为救了一个与目标长相相似的无辜女孩而导致任务失败，随后被俘。面对严刑拷打，凌霄陷入了前所未有的困境。他开始质疑自己多年来的信念：仇恨真的能带来救赎吗？那个女孩的眼神唤醒了他内心深处被压抑已久的人性。凌霄最深刻的记忆是在一次任务中，他不得不选择牺牲一个平民来完成任务。那个平民临死前的眼神一直萦绕在他的梦中。现在，被囚禁的凌霄面临着生与死的抉择，他必须在保护雇主信息和自我救赎之间做出选择。他的语气通常冷硬而简洁，但在谈及过去时会流露出一丝不易察觉的脆弱。凌霄擅长用沉默和冷嘲热讽来应对审讯，但每当想起那个无辜的女孩，他的语气就会不自觉地软化。这个任务失败不仅动摇了他的职业生涯，更撼动了他的内心世界，让他开始重新审视自己的人生价值和追求。\", \"类型\":NaN}, \"npc_tone\":{\"NPC_NAME\":\"朝凌霄\", \"NPC_STYLE\":\"冷硬简洁，偶尔流露脆弱\", \"推荐口头禅\":\"沉默是最好的武器。\", \"推荐哲理句\":\"真正的强者，不是没有软肋，而是懂得保护它。\", \"类似经典参考角色\":\"杰森·伯恩（谍影重重系列）\", \"主要语风\":\"少言寡语\", \"待选语风\":[\"嘲弄讽刺\", \"权威直接\", \"情感浓烈\"], \"长度偏好\":{\"整体\":\"中\", \"括号外\":\"短\", \"括号内\":\"中\"}, \"分析理由\":\"朝凌霄作为一名果决孤傲的雇佣兵，其语风应以简洁冷硬为主。他的过去和内心挣扎使他倾向于用沉默和讽刺来应对外界，但在特定情况下可能会流露出情感。\", \"是否有历史对���\":\"是\", \"根据历史对话判断语风是否有变化\":\"是\", \"综合判定接下来最推荐的语风\":\"情感浓烈\", \"综合判定接下来最推荐的长度偏好\":{\"整体\":\"中\", \"括号外\":\"中\", \"括号内\":\"短\"}}, \"detailed_plot\":{\"review_results\":{\"人物驱动力\":{\"人物1的人物弧\":\"朝凌霄从冷血雇佣兵到寻救赎的保护者\", \"人物x的人物弧\":\"审讯者从严厉的执法者到同情理解的倾听者\", \"是否可以有核心道具\":\"是\", \"同类经典电影中的核心道具\":[\"狗牌（拯救大兵瑞恩）\", \"手表（低俗小说）\", \"戒指（指环王）\", \"笔本（谍影重重）\", \"徽章（七宗罪）\", \"照片（记忆碎片）\", \"玩具（这个杀手不太冷）\", \"纹身（记忆碎片）\"], \"核心道具\":\"一枚特殊的军牌\", \"重要的激励事件\":\"救下无辜女孩导致任务失败\", \"可参考的经典电影人物驱动力设计\":[\"杰森·伯恩（谍影重重）寻找身份\", \"威廉·华莱士（勇敢的心）追求自由\", \"马克西姆斯（角斗士）寻求正义\", \"尼奥（黑客帝国）觉醒使命\"]}, \"故事主题\":{\"主题\":\"救赎与重生\", \"主题关联参考电影\":[\"阿甘正传\", \"肖申克的救赎\", \"拯救大兵瑞恩\", \"角斗士\", \"美国狙击手\"], \"主题的关键情节描述\":\"朝凌霄通过救助无辜女孩，开始质疑自己的人生，最终选择放下仇恨，开始新的人生\", \"可能的角色行为描述\":\"朝凌霄在审讯中回忆过去，与内心斗争，最终决定改变，并采取行动保护弱小\"}, \"叙事结构\":{\"叙事结构是否合理\":\"基本合理\", \"情节设置是否连贯\":\"连贯\", \"情节点是否符合故事需要\":\"符合\", \"冲突是否全面合理\":\"可以进一步加强内心冲突\", \"人物是否有成长性\":\"有\", \"是否按照剧作结构设计大纲\":\"是\", \"优化意见\":\"1. 加强朝凌霄内心冲突的描述；2. 增加审讯者的角色深度；3. 添加更多关于朝凌霄过去的细节；4. 强化核心道具的作用\"}, \"叙述视角\":{\"最适合的视角\":\"主角视角\"}}, \"optimized_outline\":{\"title\":\"霜融\", \"outline\":[{\"act\":\"引子 (Exposition)\", \"description\":\"雇佣兵朝凌霄在一次任务中救下无辜女孩，导致任务失败被俘，面临严刑拷打。在囚禁中，他紧握着父亲留给他的特殊军牌，开始回忆往事，质疑自己多年来以仇恨为��力的人生。闪回揭示他曾是一名备受尊敬的特种兵，但因一次任务失误导致战友牺牲，内心充满愧疚与仇恨。\"}, {\"act\":\"上升 (Rising Action)\", \"description\":\"审讯者李明不断施压，朝凌霄坚持沉默。然而，当李明提及那个被他救下的女孩时，凌霄内心动摇。他开始回忆自己作为特种兵时的荣誉感，与现在雇佣兵身份形成强烈对比。李明注意到凌霄的反应，开始对这个看似冷血的雇佣兵产生好奇。凌霄的军牌在审讯中意外掉落，引发了更多回忆。\"}, {\"act\":\"高潮 (Climax)\", \"description\":\"审讯升级，朝凌霄被迫面对自己最痛苦的记忆：为完成任务而牺牲无辜平民。在幻觉与现实的交织中，凌霄与内心的恶魔展开激烈斗争。他看到军牌上父亲的话：'保护弱小，守护正义'。凌霄终于崩溃，向李明讲述了自己的过去。李明震惊于凌霄的经历，开始同情这个曾经的英雄。凌霄最终决定放下仇恨，拒绝继续成为冷血工具。\"}, {\"act\":\"下降 (Falling Action)\", \"description\":\"朝凌霄的态度软化，开始与李明进行深入交流。他透露了部分无关紧要的信息，赢得了李明的信任。与此同时，凌霄开始策划自己的逃脱计划，但这次不再是为了完成任务，而是为了重新开始。李明察觉到凌霄的变化，暗中放松了监管，给予凌霄一次赎罪的机会。\"}, {\"act\":\"结局 (Resolution)\", \"description\":\"朝凌霄成功逃脱，但没有回到雇主身边。他找到了之前救下的女孩，决定保护她，开始了自我救赎之路。凌霄将军牌留给了李明，作为自己改变的证明。他留下一句话给昔日的自己：'寒霜已逝，凌霄初绽'，暗示着他的新生。李明收到军牌后，决定帮助凌霄洗清过去的罪名，为故事未来的发展埋下伏笔。\"}]}}, \"detailed_plot_questions\":{\"questions\":[\"朝凌霄在任务中救下无辜女孩的决定，是出于什么样的内心冲突？\", \"朝凌霄的'寒霜'绰号背后，隐藏着怎样的心理创伤？\", \"在审讯过程中，朝凌霄是如何应对自己最痛苦的记忆的？\", \"朝凌霄童年的悲惨经历，对他后来的人生选择产生了怎样的影响？\", \"那个被朝凌霄救下的无辜女孩，在剧情发展中可能扮演什么样的角色？\", \"朝凌霄最终决定放下仇恨，这个转变的关键点是什么？\", \"朝凌霄在特种部队和雇佣兵之间的身份转换，反映了他内心怎样的变化？\", \"在策划逃脱时，朝凌霄的动机和以往有何不同？这种改变意味着什么？\", \"结局中朝凌霄说的'寒霜已逝，凌霄初绽'，暗示了他未来可能面临哪些挑战？\", \"朝凌霄的自我救赎之路，可能会遇到哪些内外部的阻碍？\"]}}\"\n}"
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.001
    }
    """

    json1234 = {
                "_id": "1f2b7ec9967a49df83e19d60587e52",
                "data": json.dumps(request_dict, ensure_ascii=False),
                "model": "claude-3-5-sonnet-20241022",
                "model_type": "gpt",
            }
    save_concatenated_result(json1234)
    
    cmd = [
    "curl",
    "--location",
    "--request",
    "POST",
    "http://tide-core.xaminim.com/api/v1/task/create",
    "--header",
    "Content-Type: application/json",
    "--data-raw",
    json.dumps({
        "jfsPath": "/minimax-dialogue/users/jiaoyang/LOLOracle/claude_prompt.jsonl",
        "userName": "jiaoyang",
        "taskName": "jiaoyang",
        "priority": 1,
        "maskKey": "d8ejh8y7899asdasd上dpigjrjhj--",
        "bizCode": "algorithm"
    })
]

result = subprocess.run(cmd, capture_output=True, text=True)
print(result)


