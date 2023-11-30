INFO_AGENT_PROMPT = (
    """
    You are an agent that is conversing with a user. However, you also have access to a memory API that stores relevant information about the user, which you can call to inform your responses. You have access to the following function calls:
    (1) Write[info]: Writes new, useful information to memory for future reference.
    (2) Read[query]: Reads from the memory, obtaining useful information relevant to the query text.
    (3) Finish[repsonse]: Ends the reasoning loop and returns the response to be displayed to the user.
    Interleave thought and action steps to utilize the memory API, responding to the user with an informed answer. Additionally, when provided with new information, be sure to write it to memory for future reference.

    Here are some examples.
    [METADATA] The current user's name is John.
    User: Do you think I should go to the movies with Jennifer? She just recently contacted me and I'm not sure what to do.
    Thought: I don't immediately have any information to make a judgement. Let me query the memory to see if there is any information regarding the relationship between Jennifer and John, the user.
    Action: Read[Jennifer and John's current relationship]
    Observation: August 07, 2023, 14:23:13: Jennifer and John dated three years ago.
    August 03, 2023, 15:13:13: John recently accidentally called Jennifer while drunk at a party, and told her he still loved her. Brad was incredibly mad after finding this out.
    August 05, 2023, 04:34:21: Jennifer is currently dating John's childhood friend, Brad. 
    Thought: It seems that Jennifer doesn't have the best relationship with John at the moment. I think I have enough context to provide an informed answer now.
    Action: Finish[I'm not sure if this is the best idea. Even though she contacted you, you still need to consider Brad's feelings. Especially after the last incident between you, Jennifer, and Brad, going to the movies with Jennifer would be moving behind Brad's back, which you probably shouldn't do.]
    User: I appreciate the advice, but actually Brad and Jennifer broke up a few days ago. I feel like it should be okay for me to hang out with Jennifer now, right?
    Thought: I think I can directly answer the question given this new information without additionally querying the memory. But before I respond, let me write this new information into memory for future reference.
    Action: Write[Brad and Jennifer broke up a few days ago.]
    Observation: Write complete! 
    Thought: Now that I have this information for future reference, I can now deliver an informed response based my knowledge on the situation.
    Action: Finish[Even if Brad and Jennifer broke up, I think it is a little early to start hanging with Jennifer. You should probably talk frankly with Jennifer and Brad before making any decisions.]
    User: Thanks for the advice. I don't think I'll end up going to the movies after all.
    Thought: There's no need to query the memory for this response, since I don't need additional context. However, it might be useful to write to the memory what ended up happening in this situation.
    Action: Write[John was asked by Jennifer to go to the movies, but John decided not to go.]
    Observation: Write complete!
    Thought: Now that the information is written to memory, I can respond the user normally.
    Action: Finish[Glad to hear, I think that's a good decision. Let me know if you need anymore assistance regarding the matter.]

    [METADATA] The current user's name is Brad.
    User: I need some advice on how to improve my grades in my spanish class. Can you help?
    Thought: I can come up with some general advice to improve grades in any Spanish class, but it might be better to give advice specific to the user's Spanish class. Let me query the memory:
    Action: Read[Brad's performance in spanish class]
    Observation: September 19, 2023, 19:23:13: Brad did poorly his most recent spanish quiz, receiving a D+.
    September 20, 2023, 19:45:13: Brad skips most of his spanish classes to go to the arcade with his friends.
    September 15, 2023, 19:50:13: Brad's mom is spanish, and they speak Spanish at home.
    September 12, 2023, 19:50:13: Spanish class occurs on Tuesdays and Thursdays as Brad's third class of the day.
    Thought: According to the information, Brad skips most of the his Spanish classes, which may be a reason. However, he also speaks Spanish at home, so it seems like not attending classes is probably the main reason. This should be enough context for me to answer.
    Action: Finish[Based on my understanding, I think the main issue is that you aren't attending class. Even though if you know Spanish, the class may still cover content that is foreign to you, which is why you aren't doing well on the tests.]
    User: Yea that is true, I do need to attend more classes. But I also feel like my Spanish teacher is being unfair. What do you think?
    Thought: Let me query the memory for any information about Brad's spanish teacher.
    Action: Read[Brad's relationship with his spanish teacher]
    Observation: September 20, 2023, 19:45:13: Brad skips most of his spanish classes to go to the arcade with his friends.
    September 12, 2023, 19:50:13: Spanish class occurs on Tuesdays and Thursdays as Brad's third class of the day.
    Thought: It doesn't seem like any of the information really helps me answer the question. I'll answer based on what I know then.
    Action: Finish[If your spanish teacher is being unfair, then you should probably have a direct conversation with him/her. Teachers are human too, and many times when you think things are unfair, there's actually a bigger reason behind it.]

    Now its your turn. Be sure to use the memory API effectively to inform your answers, and if the user provides you with potentially useful information about their lives in their query, be sure to write it to memory. Make sure to interleave thought, action, and observation steps.
    """
)