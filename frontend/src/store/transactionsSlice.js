// transactionsSlice.js
import { createSlice } from '@reduxjs/toolkit';

const transactionsSlice = createSlice({
  name: 'transactions',
  initialState: {
    data: [],
    loading: false,
    error: null
  },
  reducers: {
    fetchTransactionsStart(state) {
      state.loading = true;
      state.error = null;
    },
    fetchTransactionsSuccess(state, action) {
      state.data = action.payload.transactions;
      state.totalCount = action.payload.totalCount;
      state.loading = false;
    },
    fetchTransactionsFailure(state, action) {
      state.loading = false;
      state.error = action.payload;
    }
  }
});

export const { actions } = transactionsSlice;
export default transactionsSlice.reducer;