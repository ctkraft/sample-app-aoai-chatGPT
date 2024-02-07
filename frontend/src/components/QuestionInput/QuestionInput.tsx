import React, { useState } from "react";
import { Stack, TextField } from "@fluentui/react";
import { SendRegular } from "@fluentui/react-icons";
import { MultiSelect } from "react-multi-select-component";
import Send from "../../assets/Send.svg";
import styles from "./QuestionInput.module.css";

interface Props {
    onSend: (question: string, id?: string) => void;
    disabled: boolean;
    placeholder?: string;
    clearOnSend?: boolean;
    conversationId?: string;
}

export interface Option {  
    label: string;  
    value: string;  
}  

interface AppProps {  
    onSelectedChange: (selected: Option[]) => void;  
} 

const options = [
    { label: "Congressional Budget Justifications", value: "congressional_budget_justification" },
    { label: "Questions for the Record", value: "qfr" },
    { label: "Supplemental Documents", value: "supplemental"}
];

export function Filter({ onSelectedChange }: AppProps) {
    const [selected, setSelected] = useState<Option[]>(options);

    const handleChange = (selectedOptions: Option[]) => {  
        setSelected(selectedOptions);  
        onSelectedChange(selectedOptions);  
    };  

    return (
        <div className={styles.questionInputFilter}>
            <MultiSelect
                value={selected}
                options={options}
                onChange={handleChange}
                labelledBy="Select"
                overrideStrings={{"selectSomeItems": "Select Document Types to Search", "allItemsAreSelected": "All Document Types Selected"}}
            />
        </div>
    );
};

export const QuestionInput = ({ onSend, disabled, placeholder, clearOnSend, conversationId }: Props) => {
    const [question, setQuestion] = useState<string>("");

    const sendQuestion = () => {
        if (disabled || !question.trim()) {
            return;
        }

        if(conversationId){
            onSend(question, conversationId);
        }else{
            onSend(question);
        }

        if (clearOnSend) {
            setQuestion("");
        }
    };

    const onEnterPress = (ev: React.KeyboardEvent<Element>) => {
        if (ev.key === "Enter" && !ev.shiftKey) {
            ev.preventDefault();
            sendQuestion();
        }
    };

    const onQuestionChange = (_ev: React.FormEvent<HTMLInputElement | HTMLTextAreaElement>, newValue?: string) => {
        setQuestion(newValue || "");
    };

    const sendQuestionDisabled = disabled || !question.trim();

    return (
        <Stack horizontal className={styles.questionInputContainer}>
            <TextField
                className={styles.questionInputTextArea}
                placeholder={placeholder}
                multiline
                resizable={false}
                borderless
                value={question}
                onChange={onQuestionChange}
                onKeyDown={onEnterPress}
            />
            <div className={styles.questionInputSendButtonContainer} 
                role="button" 
                tabIndex={0}
                aria-label="Ask question button"
                onClick={sendQuestion}
                onKeyDown={e => e.key === "Enter" || e.key === " " ? sendQuestion() : null}
            >
                { sendQuestionDisabled ? 
                    <SendRegular className={styles.questionInputSendButtonDisabled}/>
                    :
                    <img src={Send} className={styles.questionInputSendButton}/>
                }
            </div>
            <div className={styles.questionInputBottomBorder} />
        </Stack>
    );
};
