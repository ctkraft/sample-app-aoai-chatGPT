import React, { createContext, useReducer, ReactNode } from 'react';  
import { Option } from '../components/QuestionInput/QuestionInput'
import { appStateReducer } from './AppReducer';

export interface SelectState {
    filters: Option[]
}
const initialSelectState: SelectState = {
    filters: [
        { label: "Congressional Budget Justifications", value: "congressional_budget_justifications" },
        { label: "Questions for the Record", value: "qfr" },
        { label: "Supplemental Documents", value: "supplemental"}
    ]
} 

export type SelectAction = 
    | {type: "SET_DOC_FILTERS", payload: Option[] | null}

export const SelectedContext = createContext<{
    state: SelectState;
    dispatch: React.Dispatch<Action>;
} | undefined>(undefined);  

type AppStateProviderProps = {
    children: ReactNode;
};

export const SelectedProvider: React.FC<AppStateProviderProps> = ({ children }) => {
    const [state, dispatch] = useReducer(appStateReducer, initialSelectState)
}

SelectedContext.Provider;  
  