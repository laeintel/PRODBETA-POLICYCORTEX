declare module 'react-window-infinite-loader' {
  import { Component } from 'react';
  
  interface InfiniteLoaderProps {
    isItemLoaded: (index: number) => boolean;
    loadMoreItems: (startIndex: number, stopIndex: number) => Promise<void> | void;
    itemCount: number;
    children: (props: { onItemsRendered: any; ref: any }) => React.ReactElement;
    threshold?: number;
  }
  
  export default class InfiniteLoader extends Component<InfiniteLoaderProps> {
    resetloadMoreItemsCache(): void;
  }
}